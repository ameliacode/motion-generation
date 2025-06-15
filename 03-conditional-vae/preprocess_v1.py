import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from config import *
from fairmotion.data import bvh
from fairmotion.ops.motion import resample
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from utils.Pivots import Pivots
from utils.Quaternions import Quaternions


def get_locomotion_files(data_path="../motionsynth_data/data/processed/cmu"):
    """Get only locomotion category BVH files from CMU dataset."""
    all_files = glob.glob(os.path.join(data_path, "*.bvh"))
    locomotion_files = []

    for file_path in all_files:
        filename = os.path.basename(file_path)
        subject_num = int(filename.split("_")[0])
        if subject_num in LOCOMOTION_SUBJECTS:
            locomotion_files.append(file_path)

    return sorted(locomotion_files)


def calculate_forward_direction(motion, global_positions):
    """Calculate forward direction using shoulder and hip indices"""

    sdr_l_idx = motion.skel.get_index_joint("LeftShoulder")
    sdr_r_idx = motion.skel.get_index_joint("RightShoulder")
    hip_l_idx = motion.skel.get_index_joint("LeftUpLeg")
    hip_r_idx = motion.skel.get_index_joint("RightUpLeg")

    across1 = global_positions[:, hip_l_idx] - global_positions[:, hip_r_idx]
    across0 = global_positions[:, sdr_l_idx] - global_positions[:, sdr_r_idx]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = gaussian_filter1d(forward, 20, axis=0, mode="nearest")
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    return forward


def process_bvh_file(filepath):
    motion = bvh.load(filepath)
    if motion.fps != 120:
        return

    resample(motion, fps=30)

    num_frames = motion.num_frames() - 1
    num_joints = motion.skel.num_joints()

    global_positions = (
        motion.positions(local=False) * 0.22
    )  # (frame, joint, 3) # For scaling bvh files
    global_orientations = motion.rotations(local=False)[..., :, :2].reshape(
        -1, num_joints, 6
    )

    global_forward_vectors = global_orientations[..., :3]
    global_up_vectors = global_orientations[..., 3:]

    forward = calculate_forward_direction(motion, global_positions)
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]

    root_linear_velocity = global_positions[1:, 0:1] - global_positions[:-1, 0:1]
    local_positions = global_positions.copy()
    local_positions[:, :, 0] = global_positions[:, :, 0] - global_positions[:, 0:1, 0]
    local_positions[:, :, 2] = global_positions[:, :, 2] - global_positions[:, 0:1, 2]
    local_velocities = global_positions[1:] - global_positions[:-1]
    local_forward_vectors = global_forward_vectors.copy()
    local_up_vectors = global_up_vectors.copy()

    local_positions = rotation * local_positions
    local_velocities = rotation[1:] * local_velocities
    local_forward_vectors = rotation * local_forward_vectors
    local_up_vectors = rotation * local_up_vectors

    root_linear_velocity = rotation[1:] * root_linear_velocity
    root_angular_velocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    local_positions = local_positions[:-1]
    local_forward_vectors = local_forward_vectors[:-1]
    local_up_vectors = local_up_vectors[:-1]

    output_features = np.concatenate(
        [
            root_linear_velocity[:, :, 0],
            root_linear_velocity[:, :, 2],
            root_angular_velocity,
        ],
        axis=-1,
    )

    output_features = np.append(
        output_features, local_positions.reshape(num_frames, -1), axis=-1
    )
    output_features = np.append(
        output_features, local_velocities.reshape(num_frames, -1), axis=-1
    )
    output_features = np.append(
        output_features, local_forward_vectors.reshape(num_frames, -1), axis=-1
    )
    output_features = np.append(
        output_features, local_up_vectors.reshape(num_frames, -1), axis=-1
    )
    return output_features


def process_file_wrapper(filepath):
    """Wrapper function for multiprocessing"""
    try:
        return process_bvh_file(filepath), None
    except Exception as e:
        return None, str(e)


def main():
    if os.path.exists("./data/03_data.npz"):
        return

    bvh_files = get_locomotion_files("../motionsynth_data/data/processed/cmu")
    print(f"Found {len(bvh_files)} locomotion files")

    all_data = []
    end_indices = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(process_file_wrapper, filepath): filepath
            for filepath in bvh_files
        }

        for future in tqdm(
            as_completed(future_to_file),
            total=len(bvh_files),
            desc="Preprocessing",
        ):
            filepath = future_to_file[future]
            data, error = future.result()

            if data is not None:
                all_data.extend(data)
                end_indices.append(len(all_data) - 1)
            else:
                tqdm.write(f"Error processing {os.path.basename(filepath)}: {error}")

    np.savez("./data/03_data.npz", data=all_data, end_indices=end_indices)
    print(f"Done! {len(all_data)} frames, {len(end_indices)} sequences")

    if not os.path.exists("./data/pose0.npy"):
        pose0 = np.load("./data/03_data.npz")["data"][0]
        pose0 = np.expand_dims(pose0, axis=0)
        np.save("./data/pose0.npy", pose0)


if __name__ == "__main__":
    main()

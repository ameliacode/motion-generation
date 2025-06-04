import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from config import *
from fairmotion.core.velocity import *
from fairmotion.data import bvh
from tqdm import tqdm


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


def process_bvh_file(filepath, orient=False):
    data = []
    root_idx = 0

    motion = bvh.load(filepath)
    motion_from_velocity = MotionWithVelocity.from_motion(motion)

    frames = motion.num_frames()
    num_joints = motion.skel.num_joints()

    positions = motion.positions(local=False)
    orientations = motion.rotations(local=False)[..., :, :2].reshape(-1, num_joints, 6)

    prev_xz = [1, 0]
    for frame in range(1, frames):
        frame_data = []
        current_pos = positions[frame][root_idx]
        prev_pos = positions[frame - 1][root_idx]
        delta_x = current_pos[0] - prev_pos[0]
        delta_z = current_pos[2] - prev_pos[2]

        if orient:
            current_orient = orientations[frame][root_idx]
            prev_orient = orientations[frame - 1][root_idx]
            diff_orient = current_orient - prev_orient
            delta_facing = np.deg2rad(diff_orient[1])
        else:
            current_xz = [delta_x, delta_z]
            delta_facing = np.arctan2(current_xz[1], current_xz[0]) - np.arctan2(
                prev_xz[1], prev_xz[0]
            )
            prev_xz = current_xz

        frame_data.append(delta_x)
        frame_data.append(delta_z)
        frame_data.append(delta_facing)

        for joint in range(num_joints):
            frame_data.append(positions[frame][joint][0])
            frame_data.append(positions[frame][joint][1])
            frame_data.append(positions[frame][joint][2])

        velocity = motion_from_velocity.get_velocity_by_frame(frame)
        for joint in range(num_joints):
            frame_data.append(velocity.data_global[joint][1])
            frame_data.append(velocity.data_global[joint][3])
            frame_data.append(velocity.data_global[joint][5])

        for joint in range(num_joints):
            for index in range(6):
                frame_data.append(orientations[frame][joint][index])

        data.append(frame_data)

    return data


def process_file_wrapper(filepath):
    """Wrapper function for multiprocessing"""
    try:
        return process_bvh_file(filepath), None
    except Exception as e:
        return None, str(e)


def main():
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
            desc="Processing BVH files",
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


if __name__ == "__main__":
    main()

import math
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
from fairmotion.core.velocity import *
from fairmotion.data import bvh
from scipy.spatial.transform import *


def get_joint_names(motion):
    for idx in range(len(motion.skel.joints)):
        print(idx, motion.skel.joints[idx].name)


def generate_mocap(root_idx=0, folder_name=None, file_name=None, orient=False):
    data = []
    current_dir = Path(__file__).resolve().parents[0]

    motion = bvh.load(str(current_dir / "data" / "bvh" / folder_name / file_name))
    motion_from_velocity = MotionWithVelocity.from_motion(motion)
    # motion_from_velocity.compute_velocities()
    get_joint_names(motion)

    frames = motion.num_frames()
    num_joints = motion.skel.num_joints()

    positions = (
        motion.positions(local=False) * 0.22
    )  # (frame, joint, 3) # For scaling bvh files
    orientations = motion.rotations(local=False)[..., :, :2].reshape(-1, num_joints, 6)

    start_index = 502  # 0
    end_index = 7489 + 1  # frames
    prev_xz = [1, 0]
    for frame in range(start_index, end_index):  # Discard T pose
        frame_data = []
        # root displacement
        current_pos = positions[frame][root_idx]  # root current position
        prev_pos = positions[frame - 1][root_idx]  # root previous position
        delta_x = current_pos[0] - prev_pos[0]
        delta_z = current_pos[2] - prev_pos[2]  # Y axis up, ground projection

        if orient:  # utilize orientation information from bvh
            current_orient = orientations[frame][root_idx]
            prev_orient = orientations[frame - 1][root_idx]
            diff_orient = current_orient - prev_orient
            delta_facing = np.deg2rad(diff_orient[1])
        else:
            # root delta and its displacements
            current_xz = [delta_x, delta_z]
            delta_facing = np.arctan2(current_xz[1], current_xz[0]) - np.arctan2(
                prev_xz[1], prev_xz[0]
            )  # angle between displacement
            prev_xz = current_xz

        frame_data.append(delta_x)
        frame_data.append(delta_z)
        frame_data.append(delta_facing)

        # joint coordinates
        for joint in range(num_joints):
            frame_data.append(positions[frame][joint][0])
            frame_data.append(positions[frame][joint][1])
            frame_data.append(positions[frame][joint][2])

        # joint velocities in Cartesian coordinate in root frame
        velocity = motion_from_velocity.get_velocity_by_frame(frame)
        for joint in range(num_joints):
            frame_data.append(velocity.data_global[joint][1])  # LINEAR VELOCITY
            frame_data.append(velocity.data_global[joint][3])
            frame_data.append(velocity.data_global[joint][5])

        # 6D joint orientations (first two columns of rotation matrix)
        for joint in range(num_joints):
            for index in range(6):
                frame_data.append(orientations[frame][joint][index])

        if frame == 10:
            np.save(str(current_dir / "data" / "pose0.npy"), [frame_data])

        data.append(frame_data)

    # end_indices = frames - 1
    # end_indices = np.cumsum([502, 1365, 2289, 3112, 3935, 4759, 5582, 6425, 7489, frames]) - 1 # Testing for HDM05
    # end_indices = np.array([502, 1365, 2289, 3112, 3935, 4759, 5582, 6425, 7489, frames-1])
    # end_indices = np.array([863, 1787, 2610, 3433, 4257, 5080, 5923, 6988])
    end_indices = 6988
    np.savez(
        str(current_dir / "data" / "mocap.npz"), data=data, end_indices=end_indices
    )


def test_load_npz():
    current_dir = Path(__file__).resolve().parents[0]
    read_npz = np.load(str(current_dir / "data" / "mocap.npz"))
    print(read_npz.files)
    print(len(read_npz["data"]))
    print(read_npz["end_indices"])


def test_load_npy():
    data = []
    current_dir = Path(__file__).resolve().parents[0]
    pose0 = np.load(str(current_dir / "data" / "test-pose0.npy"))
    print(len(pose0))


def main():
    generate_mocap(
        folder_name="HDM05", file_name="HDM_mm_03-02_01_120.bvh", orient=True
    )
    test_load_npz()
    # test_load_npy()


if __name__ == "__main__":
    main()

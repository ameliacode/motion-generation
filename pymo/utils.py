import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


def extract_positions(track):
    """Extract positions and return joint names mapping"""
    position_cols = [c for c in track.values.columns if "position" in c]
    joints = sorted(list(set([c.split("_")[0] for c in position_cols])))

    positions = []
    for frame_idx in range(len(track.values)):
        frame_positions = []
        for joint in joints:
            x = track.values[f"{joint}_Xposition"].iloc[frame_idx]
            y = track.values[f"{joint}_Yposition"].iloc[frame_idx]
            z = track.values[f"{joint}_Zposition"].iloc[frame_idx]
            frame_positions.append([x, y, z])
        positions.append(frame_positions)

    return np.array(positions), joints


def find_joint_index(joint_names, target_name):
    """Find joint index from list of possible names"""
    if target_name in joint_names:
        return joint_names.index(target_name)
    return None


def calculate_forward_direction(positions, joint_names):
    """Calculate forward direction using shoulder and hip indices"""

    sdr_l_idx = find_joint_index(joint_names, "LeftShoulder")
    sdr_r_idx = find_joint_index(
        joint_names,
        "RightShoulder",
    )
    hip_l_idx = find_joint_index(joint_names, "LeftUpLeg")
    hip_r_idx = find_joint_index(
        joint_names,
        "RightUpLeg",
    )

    across1 = positions[:, hip_l_idx] - positions[:, hip_r_idx]
    across0 = positions[:, sdr_l_idx] - positions[:, sdr_r_idx]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = gaussian_filter1d(forward, 20, axis=0, mode="nearest")
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

    return forward


def update_positions_in_dataframe(df, positions, joint_names):
    """Update dataframe with transformed positions"""
    for i, joint in enumerate(joint_names):
        if i < positions.shape[1]:
            if f"{joint}_Xposition" in df.columns:
                df[f"{joint}_Xposition"] = positions[:, i, 0]
            if f"{joint}_Yposition" in df.columns:
                df[f"{joint}_Yposition"] = positions[:, i, 1]
            if f"{joint}_Zposition" in df.columns:
                df[f"{joint}_Zposition"] = positions[:, i, 2]


def update_velocities_in_dataframe(df, velocities, joint_names):
    """Update dataframe with joint velocities."""
    for i, joint_name in enumerate(joint_names):
        df[f"{joint_name}_Xvelocity"] = velocities[:, i, 0]
        df[f"{joint_name}_Yvelocity"] = velocities[:, i, 1]
        df[f"{joint_name}_Zvelocity"] = velocities[:, i, 2]


def update_orientations_in_dataframe(df, orientations, joint_names):
    """Update dataframe with joint orientations (forward/up vectors)."""
    for i, joint_name in enumerate(joint_names):
        df[f"{joint_name}_forward_X"] = orientations[:, i, 0]
        df[f"{joint_name}_forward_Y"] = orientations[:, i, 1]
        df[f"{joint_name}_forward_Z"] = orientations[:, i, 2]

        df[f"{joint_name}_up_X"] = orientations[:, i, 3]
        df[f"{joint_name}_up_Y"] = orientations[:, i, 4]
        df[f"{joint_name}_up_Z"] = orientations[:, i, 5]

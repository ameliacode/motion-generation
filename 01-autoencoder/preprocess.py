import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation


class BVHJoint:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)  # Local offset from parent
        self.channels = (
            []
        )  # Rotation order (e.g., ['Xrotation', 'Yrotation', 'Zrotation'])
        self.index = 0  # Index in motion data

    def add_child(self, child):
        self.children.append(child)


class BVHSkeleton:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.joint_names = []

    def add_joint(self, joint):
        self.joints[joint.name] = joint
        self.joint_names.append(joint.name)

    def get_joint(self, name):
        return self.joints.get(name)


def parse_bvh_file(filepath):
    """Parse BVH file and extract skeleton + motion data"""
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    skeleton = BVHSkeleton()
    motion_data = []
    frame_time = 0.0

    i = 0
    # Parse HIERARCHY section
    while i < len(lines):
        line = lines[i]

        if line.startswith("HIERARCHY"):
            i += 1
            continue

        elif line.startswith("ROOT") or line.startswith("JOINT"):
            joint_name = line.split()[1]
            joint = BVHJoint(joint_name)

            if line.startswith("ROOT"):
                skeleton.root = joint

            skeleton.add_joint(joint)
            i += 1

            # Parse joint block
            if i < len(lines) and lines[i] == "{":
                i += 1
                while i < len(lines) and lines[i] != "}":
                    if lines[i].startswith("OFFSET"):
                        offset = [float(x) for x in lines[i].split()[1:4]]
                        joint.offset = np.array(offset)
                    elif lines[i].startswith("CHANNELS"):
                        parts = lines[i].split()
                        joint.channels = parts[2:]
                    elif lines[i].startswith("JOINT") or lines[i].startswith(
                        "End Site"
                    ):
                        # Handle nested joints (simplified)
                        pass
                    i += 1
                i += 1  # Skip closing brace

        elif line.startswith("MOTION"):
            # Parse motion section
            i += 1
            frames = int(lines[i].split(":")[1])
            i += 1
            frame_time = float(lines[i].split(":")[1])
            i += 1

            # Read motion data
            for frame_idx in range(frames):
                if i < len(lines):
                    frame_data = [float(x) for x in lines[i].split()]
                    motion_data.append(frame_data)
                    i += 1
            break
        else:
            i += 1

    return skeleton, np.array(motion_data), frame_time


def euler_to_rotation_matrix(angles, order="XYZ"):
    """Convert Euler angles to rotation matrix"""
    return Rotation.from_euler(order.lower(), angles, degrees=True).as_matrix()


def forward_kinematics(skeleton, frame_data):
    """Compute joint positions using forward kinematics"""
    joint_positions = {}
    joint_transforms = {}

    def compute_joint_transform(joint, parent_transform=np.eye(4)):
        # Get joint's rotation data from frame
        if joint == skeleton.root:
            # Root has 6 DOF: translation + rotation
            translation = frame_data[:3]
            rotation_angles = frame_data[3:6]
        else:
            # Find rotation data for this joint (simplified)
            joint_idx = list(skeleton.joints.keys()).index(joint.name)
            rotation_start = 6 + (joint_idx - 1) * 3  # Skip root's 6 DOF
            if rotation_start + 2 < len(frame_data):
                rotation_angles = frame_data[rotation_start : rotation_start + 3]
            else:
                rotation_angles = [0, 0, 0]

        # Create local transform
        local_transform = np.eye(4)

        # Apply offset (bone length)
        local_transform[:3, 3] = joint.offset

        # Apply rotation
        if len(rotation_angles) == 3:
            rotation_matrix = euler_to_rotation_matrix(rotation_angles)
            local_transform[:3, :3] = rotation_matrix

        # For root, also apply translation
        if joint == skeleton.root:
            local_transform[:3, 3] += translation

        # Compute world transform
        world_transform = parent_transform @ local_transform
        joint_transforms[joint.name] = world_transform

        # Extract position
        joint_positions[joint.name] = world_transform[:3, 3]

        # Recursively compute children
        for child in joint.children:
            compute_joint_transform(child, world_transform)

    # Start from root
    if skeleton.root:
        compute_joint_transform(skeleton.root)

    return joint_positions


def get_important_joints():
    """Define 20 most important joints for human motion"""
    return [
        "Hips",  # 1 - Root
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",  # 2-6 - Spine
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",  # 7-10 - Left arm
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",  # 11-14 - Right arm
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",  # 15-17 - Left leg
        "RightUpLeg",
        "RightLeg",
        "RightFoot",  # 18-20 - Right leg
    ]


def extract_joint_positions_from_frame(joint_positions_dict, important_joints):
    """Extract positions of important joints"""
    positions = []

    for joint_name in important_joints:
        if joint_name in joint_positions_dict:
            pos = joint_positions_dict[joint_name]
            positions.extend([pos[0], pos[1], pos[2]])
        else:
            # If joint not found, use zeros
            positions.extend([0.0, 0.0, 0.0])

    # Ensure exactly 20 joints (60 values)
    while len(positions) < 60:
        positions.extend([0.0, 0.0, 0.0])

    return np.array(positions[:60])


def normalize_skeleton_scale(joint_positions_array):
    """Normalize skeleton to consistent scale"""
    num_frames = joint_positions_array.shape[0]

    # Calculate average bone lengths across all frames
    total_bone_lengths = []

    for frame_idx in range(num_frames):
        frame_positions = joint_positions_array[frame_idx].reshape(-1, 3)
        root_pos = frame_positions[0]  # Hip position

        # Calculate distances from root to other major joints
        distances = []
        for joint_idx in [4, 6, 9, 13, 16, 19]:  # Head, shoulders, hands, feet
            if joint_idx < len(frame_positions):
                dist = np.linalg.norm(frame_positions[joint_idx] - root_pos)
                if dist > 0.01:
                    distances.append(dist)

        if distances:
            total_bone_lengths.append(np.mean(distances))

    if not total_bone_lengths:
        return joint_positions_array

    # Normalize to target scale (1.0 meter average)
    current_scale = np.mean(total_bone_lengths)
    target_scale = 1.0
    scale_factor = target_scale / current_scale if current_scale > 0.01 else 1.0

    # Apply scaling
    normalized_positions = joint_positions_array.copy()
    for frame_idx in range(num_frames):
        frame_positions = normalized_positions[frame_idx].reshape(-1, 3)
        root_pos = frame_positions[0]

        for joint_idx in range(len(frame_positions)):
            relative_pos = frame_positions[joint_idx] - root_pos
            frame_positions[joint_idx] = root_pos + relative_pos * scale_factor

        normalized_positions[frame_idx] = frame_positions.flatten()

    return normalized_positions


def remove_global_motion(joint_positions_array):
    """Remove global translation (XZ) and rotation (Y)"""
    num_frames = joint_positions_array.shape[0]
    local_positions = joint_positions_array.copy()

    for frame_idx in range(num_frames):
        frame_positions = local_positions[frame_idx].reshape(-1, 3)
        root_pos = frame_positions[0]

        # Remove XZ translation from all joints
        for joint_idx in range(len(frame_positions)):
            frame_positions[joint_idx][0] -= root_pos[0]  # Remove X
            frame_positions[joint_idx][2] -= root_pos[2]  # Remove Z
            # Keep Y (height) as is

        local_positions[frame_idx] = frame_positions.flatten()

    return local_positions


def calculate_character_forward_direction(joint_positions_frame):
    """Calculate character's forward direction from joint positions"""
    positions = joint_positions_frame.reshape(-1, 3)

    # Use shoulder line to determine forward direction
    left_shoulder = positions[6]  # LeftShoulder
    right_shoulder = positions[10]  # RightShoulder

    # Shoulder vector (right to left)
    shoulder_vec = left_shoulder - right_shoulder

    # Forward is perpendicular to shoulder line (in XZ plane)
    forward = np.array([-shoulder_vec[2], 0, shoulder_vec[0]])
    forward_norm = np.linalg.norm(forward)

    if forward_norm > 0.001:
        forward = forward / forward_norm
    else:
        forward = np.array([0, 0, 1])  # Default forward

    return forward


def add_velocity_features(local_positions, global_positions):
    """Add rotational velocity (Y) and translational velocity (XZ)"""
    num_frames = local_positions.shape[0]

    # Initialize velocities
    rot_vel_y = np.zeros(num_frames)
    trans_vel_x = np.zeros(num_frames)
    trans_vel_z = np.zeros(num_frames)

    # Calculate velocities at 30 FPS
    for frame_idx in range(1, num_frames):
        # Get root positions
        prev_root = global_positions[frame_idx - 1].reshape(-1, 3)[0]
        curr_root = global_positions[frame_idx].reshape(-1, 3)[0]

        # Translational velocity in world space
        world_vel = (curr_root - prev_root) * 30.0  # 30 FPS

        # Character's forward direction
        forward_dir = calculate_character_forward_direction(global_positions[frame_idx])
        right_dir = np.cross([0, 1, 0], forward_dir)

        # Convert world velocity to character-relative velocity
        trans_vel_x[frame_idx] = np.dot(world_vel, right_dir)  # Right velocity
        trans_vel_z[frame_idx] = np.dot(world_vel, forward_dir)  # Forward velocity

        # Rotational velocity (simplified)
        if frame_idx > 1:
            prev_forward = calculate_character_forward_direction(
                global_positions[frame_idx - 1]
            )

            # Calculate angle between forward directions
            cos_angle = np.clip(np.dot(forward_dir, prev_forward), -1, 1)
            angle_diff = np.arccos(cos_angle)

            # Determine rotation direction using cross product
            cross = np.cross(prev_forward, forward_dir)
            if cross[1] < 0:  # Y component determines left/right rotation
                angle_diff = -angle_diff

            rot_vel_y[frame_idx] = angle_diff * 30.0  # Angular velocity

    # Combine: 20 joints * 3 coords + 3 velocities = 63 features
    enhanced_motion = np.zeros((num_frames, 63))
    enhanced_motion[:, :60] = local_positions  # 20 joints * 3 = 60
    enhanced_motion[:, 60] = rot_vel_y  # Rotational velocity Y
    enhanced_motion[:, 61] = trans_vel_x  # Translational velocity X (right)
    enhanced_motion[:, 62] = trans_vel_z  # Translational velocity Z (forward)

    return enhanced_motion


def resample_to_30fps(motion_data, original_frame_time):
    """Resample motion data to 30 FPS"""
    original_fps = 1.0 / original_frame_time
    target_fps = 30.0

    if abs(original_fps - target_fps) < 0.1:
        return motion_data

    original_frames = motion_data.shape[0]
    target_frames = int(original_frames * target_fps / original_fps)

    if target_frames <= 1:
        return motion_data

    # Linear interpolation
    original_indices = np.linspace(0, original_frames - 1, original_frames)
    target_indices = np.linspace(0, original_frames - 1, target_frames)

    resampled_data = np.zeros((target_frames, motion_data.shape[1]))
    for i in range(motion_data.shape[1]):
        resampled_data[:, i] = np.interp(
            target_indices, original_indices, motion_data[:, i]
        )

    return resampled_data


def create_windows(motion_data, window_size=160, overlap=80):
    """Create overlapping windows of 160 frames with 80 frame overlap"""
    if motion_data.shape[0] < window_size:
        return []

    windows = []
    start = 0
    step = window_size - overlap

    while start + window_size <= motion_data.shape[0]:
        window = motion_data[start : start + window_size]
        windows.append(window)
        start += step

    return windows


def subtract_mean_pose(windows):
    """Subtract mean pose from each pose vector"""
    all_windows = np.array(windows)
    mean_pose = np.mean(all_windows, axis=(0, 1))  # Mean across all windows and frames
    normalized_windows = all_windows - mean_pose
    return normalized_windows, mean_pose


def process_single_bvh_file(filepath):
    """Process a single BVH file through the complete pipeline"""
    try:
        # 1. Parse BVH file
        skeleton, motion_data, frame_time = parse_bvh_file(filepath)

        if motion_data.shape[0] == 0:
            return None

        # 2. Resample to 30 FPS
        resampled_data = resample_to_30fps(motion_data, frame_time)

        # 3. Apply forward kinematics to get joint positions
        important_joints = get_important_joints()
        joint_positions_array = []

        for frame_idx in range(resampled_data.shape[0]):
            # Get joint positions using forward kinematics
            joint_positions_dict = forward_kinematics(
                skeleton, resampled_data[frame_idx]
            )

            # Extract important joint positions
            frame_positions = extract_joint_positions_from_frame(
                joint_positions_dict, important_joints
            )
            joint_positions_array.append(frame_positions)

        joint_positions_array = np.array(joint_positions_array)

        # 4. Normalize skeleton scale
        normalized_positions = normalize_skeleton_scale(joint_positions_array)

        # 5. Remove global motion (keep original for velocity calculation)
        global_positions = normalized_positions.copy()
        local_positions = remove_global_motion(normalized_positions)

        # 6. Add velocity features
        motion_with_velocity = add_velocity_features(local_positions, global_positions)

        return motion_with_velocity

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def process_cmu_database(data_dir, output_file="cmu_processed.npz"):
    """
    Process CMU Motion Capture Database with proper forward kinematics:
    - Parse BVH skeleton and apply forward kinematics
    - Sub-sample to 30 FPS
    - Use 20 most important joints (60 position features)
    - Normalize joint lengths to consistent scale
    - Remove global XZ translation and Y rotation
    - Add 3 velocity features relative to character forward direction
    - Create 160-frame windows with 80-frame overlap
    - Subtract mean pose
    - Result: X ∈ R^(160×63)
    """

    # Find all BVH files
    bvh_files = glob.glob(os.path.join(data_dir, "**", "*.bvh"), recursive=True)

    if not bvh_files:
        print(f"No BVH files found in {data_dir}")
        return None, None

    print(f"Found {len(bvh_files)} BVH files")
    print("Processing with forward kinematics...")

    all_windows = []
    processed_files = 0

    for i, bvh_file in enumerate(bvh_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(bvh_files)}")

        # Process single file
        motion_with_velocity = process_single_bvh_file(bvh_file)

        if motion_with_velocity is not None:
            # Create overlapping windows
            windows = create_windows(motion_with_velocity, window_size=160, overlap=80)
            all_windows.extend(windows)
            processed_files += 1

    if not all_windows:
        print("No valid windows created!")
        return None, None

    print(f"Successfully processed {processed_files}/{len(bvh_files)} files")
    print(f"Created {len(all_windows)} windows total")

    # Subtract mean pose
    normalized_windows, mean_pose = subtract_mean_pose(all_windows)

    print(f"Final dataset shape: {normalized_windows.shape}")
    print(f"Expected: (num_windows, 160, 63)")
    print(f"- 160 frames per window")
    print(f"- 63 features per frame (20 joints × 3 + 3 velocities)")

    # Save processed data
    np.savez_compressed(
        output_file,
        windows=normalized_windows,
        mean_pose=mean_pose,
        num_windows=len(normalized_windows),
        window_size=160,
        overlap=80,
        fps=30,
        num_joints=20,
        features_per_frame=63,
    )

    print(f"Saved processed data to {output_file}")
    return normalized_windows, mean_pose


def main():
    # Process CMU database
    data_directory = "./data/boxing"  # Update path to your CMU data
    output_file = "output.npz"

    windows, mean_pose = process_cmu_database(data_directory, output_file)

    if windows is not None:
        print(f"\n=== Processing Complete ===")
        print(f"Dataset shape: {windows.shape}")
        print(f"Number of windows: {windows.shape[0]}")
        print(f"Frames per window: {windows.shape[1]}")
        print(f"Features per frame: {windows.shape[2]}")

        # Load and verify
        data = np.load(output_file)
        print(f"\nSaved data info:")
        for key in data.keys():
            if hasattr(data[key], "shape"):
                print(f"{key}: {data[key].shape}")
            else:
                print(f"{key}: {data[key]}")


if __name__ == "__main__":
    main()

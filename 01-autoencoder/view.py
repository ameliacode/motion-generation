import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def load_and_view_motion(data_file="./output.npz", sequence_idx=0):
    """Load and visualize BVH motion data"""

    # Load data
    data = np.load(data_file)
    X = data["windows"]  # shape: (N, T, D)
    mean_pose = data["mean_pose"]

    print(f"Data shape: {X.shape}")
    print(f"Number of sequences: {X.shape[0]}")
    print(f"Frames per sequence: {X.shape[1]}")
    print(f"Features per frame: {X.shape[2]}")

    # Select sequence to visualize
    if sequence_idx >= X.shape[0]:
        sequence_idx = 0
        print(f"Sequence index too high, using sequence 0")

    sequence = X[sequence_idx]  # shape: (T, D)
    T = sequence.shape[0]

    # Extract only joint positions (first 60 features = 20 joints * 3)
    joint_positions = sequence[:, :60]  # Ignore velocity features
    joint_positions = joint_positions.reshape(T, 20, 3)

    # Calculate plot bounds from actual data
    all_points = joint_positions.reshape(-1, 3)
    x_range = [all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5]
    y_range = [all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5]
    z_range = [all_points[:, 2].min() - 0.5, all_points[:, 2].max() + 0.5]

    print(f"Plot bounds:")
    print(f"X: {x_range[0]:.2f} to {x_range[1]:.2f}")
    print(f"Y: {y_range[0]:.2f} to {y_range[1]:.2f}")
    print(f"Z: {z_range[0]:.2f} to {z_range[1]:.2f}")

    # Simple skeleton connections for 20 joints
    # Based on typical human skeleton hierarchy
    connections = [
        # Spine chain: Hip -> Spine -> Spine1 -> Spine2 -> Neck -> Head
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        # Left arm: Spine2 -> LeftShoulder -> LeftArm -> LeftForeArm -> LeftHand
        (3, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        # Right arm: Spine2 -> RightShoulder -> RightArm -> RightForeArm -> RightHand
        (3, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        # Left leg: Hip -> LeftUpLeg -> LeftLeg -> LeftFoot
        (0, 14),
        (14, 15),
        (15, 16),
        # Right leg: Hip -> RightUpLeg -> RightLeg -> RightFoot
        (0, 17),
        (17, 18),
        (18, 19),
    ]

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Initialize plot elements
    points_plot = ax.scatter([], [], [], s=50, c="red", alpha=0.8)
    lines = []
    for _ in connections:
        (line,) = ax.plot([], [], [], "b-", linewidth=2)
        lines.append(line)

    # Set up the plot
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Motion Sequence {sequence_idx} - Frame 0/{T-1}")

    # Better viewing angle
    ax.view_init(elev=20, azim=45)

    def update_frame(frame):
        # Get current frame data
        points = joint_positions[frame]

        # Update scatter plot
        points_plot._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

        # Update skeleton lines
        for i, (start, end) in enumerate(connections):
            if start < len(points) and end < len(points):
                x_data = [points[start, 0], points[end, 0]]
                y_data = [points[start, 1], points[end, 1]]
                z_data = [points[start, 2], points[end, 2]]

                lines[i].set_data(x_data, y_data)
                lines[i].set_3d_properties(z_data)

        # Update title with frame number
        ax.set_title(f"Motion Sequence {sequence_idx} - Frame {frame}/{T-1}")

        return [points_plot] + lines

    # Create animation
    print(f"Creating animation with {T} frames...")
    ani = FuncAnimation(
        fig, update_frame, frames=T, interval=50, blit=False, repeat=True
    )

    # Show plot
    plt.tight_layout()
    plt.show()

    return ani


def view_multiple_sequences(data_file="./output.npz", num_sequences=3):
    """View multiple sequences side by side"""

    data = np.load(data_file)
    X = data["windows"]

    num_sequences = min(num_sequences, X.shape[0])

    fig, axes = plt.subplots(
        1,
        num_sequences,
        figsize=(5 * num_sequences, 5),
        subplot_kw={"projection": "3d"},
    )

    if num_sequences == 1:
        axes = [axes]

    for i in range(num_sequences):
        sequence = X[i][:, :60].reshape(X.shape[1], 20, 3)

        # Plot first frame of each sequence
        points = sequence[0]

        ax = axes[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=30, c="red")

        # Set equal aspect ratio
        all_points = sequence.reshape(-1, 3)
        center = all_points.mean(axis=0)
        max_range = np.abs(all_points - center).max()

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_title(f"Sequence {i}")
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the viewer"""
    import sys

    data_file = "./output.npz"

    # Check if file exists
    try:
        data = np.load(data_file)
        print(f"Loaded data from {data_file}")
        print(f"Available keys: {list(data.keys())}")
    except FileNotFoundError:
        print(f"File {data_file} not found!")
        print("Make sure you have run the BVH processing script first.")
        return

    # Ask user which sequence to view
    num_sequences = data["windows"].shape[0]
    print(f"\nAvailable sequences: 0 to {num_sequences-1}")

    try:
        if len(sys.argv) > 1:
            sequence_idx = int(sys.argv[1])
        else:
            sequence_idx = int(
                input(f"Enter sequence number to view (0-{num_sequences-1}): ")
            )
    except (ValueError, KeyboardInterrupt):
        sequence_idx = 0
        print("Using sequence 0")

    # Load and view the motion
    print(f"\nViewing sequence {sequence_idx}...")
    print("Close the plot window to exit.")

    ani = load_and_view_motion(data_file, sequence_idx)


if __name__ == "__main__":
    main()

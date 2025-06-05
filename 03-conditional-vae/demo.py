import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Generate poses from CVAE
def generate_poses(decoder, initial_pose, num_frames=20):
    poses = [initial_pose]
    current_pose = initial_pose

    for _ in range(num_frames - 1):
        z = tf.random.normal((1, LATENT_DIM))
        next_pose = decoder([z, tf.expand_dims(current_pose, 0)])
        next_pose = tf.squeeze(next_pose, 0).numpy()
        poses.append(next_pose)
        current_pose = next_pose

    return np.array(poses)


# Plot poses
def plot_poses(poses, num_cols=3):
    num_frames = len(poses)
    num_rows = (num_frames + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if num_frames == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_frames):
        pose = poses[i]
        x = pose[::2]  # X coordinates
        y = pose[1::2]  # Y coordinates

        axes[i].plot(x, y, "bo-", linewidth=2, markersize=6)
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].set_title(f"Frame {i+1}")
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(num_frames, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


# Plot trajectories
def plot_trajectories(poses, joints=[0, 1, 2]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for joint in joints:
        x_traj = poses[:, joint * 2]
        y_traj = poses[:, joint * 2 + 1]

        ax1.plot(x_traj, label=f"Joint {joint+1}", marker="o", markersize=4)
        ax2.plot(y_traj, label=f"Joint {joint+1}", marker="o", markersize=4)

    ax1.set_title("X Trajectories")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("X Position")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Y Trajectories")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Y Position")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Single pose plot
def plot_single_pose(pose, title="Pose"):
    x = pose[::2]
    y = pose[1::2]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, "bo-", linewidth=3, markersize=8)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

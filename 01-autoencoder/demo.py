import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from model import autoencoder
from mpl_toolkits.mplot3d import Axes3D

# Load data and model
data = np.load("./data/cmu_data.npz")
X = data["windows"].astype(np.float32)
autoencoder.load_weights("./01-autoencoder/weights.h5")

# Select sample to animate
sample_idx = 0
x_orig = X[sample_idx]
x_recon = autoencoder.predict(x_orig[np.newaxis, ...])[0]

# Number of joints
num_joints = 21
bones = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (7, 8),
    (8, 9),
    (2, 10),
    (10, 11),
    (11, 12),
    (2, 13),
    (13, 14),
    (14, 15),
    (0, 16),
    (16, 17),
    (17, 18),
    (0, 19),
    (19, 20),
]


def get_joint_positions(frame):
    # Convert flat (63,) to (21,3)
    return frame.reshape((num_joints, 3))


# Setup figure and 3D axes
fig = plt.figure(figsize=(10, 5))

ax_orig = fig.add_subplot(121, projection="3d")
ax_orig.set_title("Original Motion")
ax_recon = fig.add_subplot(122, projection="3d")
ax_recon.set_title("Reconstructed Motion")

# Set axis limits (adjust based on data scale)
lims = [-1, 1]
for ax in [ax_orig, ax_recon]:
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)
    ax.view_init(elev=15, azim=70)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")  # Usually Z is up in mocap
    ax.set_zlabel("Y")

# Initialize plots for bones
lines_orig = [ax_orig.plot([], [], [], "o-", color="blue")[0] for _ in bones]
lines_recon = [ax_recon.plot([], [], [], "o-", color="red")[0] for _ in bones]


def update(frame_idx):
    joints_orig = get_joint_positions(x_orig[frame_idx])
    joints_recon = get_joint_positions(x_recon[frame_idx])

    for line, (p, c) in zip(lines_orig, bones):
        xs = [joints_orig[p, 0], joints_orig[c, 0]]
        ys = [joints_orig[p, 2], joints_orig[c, 2]]  # Z is vertical
        zs = [joints_orig[p, 1], joints_orig[c, 1]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    for line, (p, c) in zip(lines_recon, bones):
        xs = [joints_recon[p, 0], joints_recon[c, 0]]
        ys = [joints_recon[p, 2], joints_recon[c, 2]]
        zs = [joints_recon[p, 1], joints_recon[c, 1]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)

    return lines_orig + lines_recon


anim = FuncAnimation(fig, update, frames=x_orig.shape[0], interval=33, blit=True)

plt.show()

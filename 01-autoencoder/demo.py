import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from config import *
from matplotlib.animation import FuncAnimation
from model import autoencoder
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Load data and model
data = np.load("./data/01_data.npz")
X = data["clips"].astype(np.float32)
Xmean = data["mean"].astype(np.float32)
Xstd = data["std"].astype(np.float32)

autoencoder.load_weights("./01-autoencoder/01_weights.h5")

# Select sample to animate
sample_idx = 0
x_orig = X[sample_idx]
x_recon = autoencoder.predict(x_orig[np.newaxis, ...])[0]
x_recon = (x_recon * Xstd) + Xmean


def animate_plot(animations):
    # Process data
    processed_animations = []
    for anim in animations:
        joints = anim[:, :-3].reshape((160, 20, 3))
        root_x, root_z, root_r = anim[:, -3], anim[:, -2], anim[:, -1]

        rotation = R.identity()
        translation = np.array([0.0, 0.0, 0.0])

        for i in range(len(joints)):
            joints[i] = rotation.apply(joints[i])
            joints[i, :, 0] += translation[0]
            joints[i, :, 2] += translation[2]

            delta_rot = R.from_rotvec(-root_r[i] * np.array([0, 1, 0]))
            rotation = delta_rot * rotation
            translation = translation + rotation.apply(
                np.array([root_x[i], 0, root_z[i]])
            )

        processed_animations.append(joints)

    # Set up plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Get bounds
    all_joints = np.concatenate(processed_animations, axis=0)
    xs, ys, zs = all_joints[:, :, 0], all_joints[:, :, 1], all_joints[:, :, 2]
    ax.set_xlim(xs.min() - 1, xs.max() + 1)
    ax.set_ylim(ys.min() - 1, ys.max() + 1)
    ax.set_zlim(zs.min() - 1, zs.max() + 1)

    ax.view_init(elev=45, azim=0, roll=90)
    # Create scatter plots and bone lines
    scatters = []
    bone_lines = []

    for joints in processed_animations:
        # Joint points
        scatter = ax.scatter([], [], [], s=10, c="red")
        scatters.append(scatter)

        # Bone connections
        lines = []
        for bone in BONES:
            (line,) = ax.plot([], [], [], "b-", linewidth=2)
            lines.append(line)
        bone_lines.append(lines)

    # Animation function
    def update(frame):
        for i, joints in enumerate(processed_animations):
            # Update joint positions
            scatters[i]._offsets3d = (
                joints[frame, :, 0],
                joints[frame, :, 1],
                joints[frame, :, 2],
            )

            # Update bone connections
            for j, (start_joint, end_joint) in enumerate(BONES):
                if start_joint < joints.shape[1] and end_joint < joints.shape[1]:
                    x_data = [
                        joints[frame, start_joint, 0],
                        joints[frame, end_joint, 0],
                    ]
                    y_data = [
                        joints[frame, start_joint, 1],
                        joints[frame, end_joint, 1],
                    ]
                    z_data = [
                        joints[frame, start_joint, 2],
                        joints[frame, end_joint, 2],
                    ]

                    bone_lines[i][j].set_data(x_data, y_data)
                    bone_lines[i][j].set_3d_properties(z_data)

        ax.set_title(f"Frame {frame}")
        return scatters + [line for lines in bone_lines for line in lines]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=160, interval=50, repeat=True)
    return ani


plt.show()

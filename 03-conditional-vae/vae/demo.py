import os
import sys

sys.path.append(os.path.join(os.getcwd(), "03-conditional-vae"))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model import cvae
from mpl_toolkits.mplot3d import Axes3D

bones = [
    [9, 10],
    [8, 9],
    [7, 8],
    [6, 7],
    [0, 6],
    [4, 5],
    [3, 4],
    [2, 3],
    [1, 2],
    [0, 1],
    [14, 24],
    [24, 25],
    [25, 26],
    [26, 27],
    [27, 28],
    [28, 29],
    [29, 30],
    [14, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [20, 21],
    [21, 22],
    [22, 23],
    [12, 13],
    [11, 12],
    [0, 11],
]


def extract_joints_xyz(v, x_ind, y_ind, z_ind):
    x = v[x_ind]
    y = v[z_ind]
    z = v[y_ind]
    return x, y, z


def animate_cvae():
    initial_pose = np.random.normal(0, 0.1, 375)
    poses = []
    prev_pose = initial_pose
    curr_pose = initial_pose

    for i in range(30):
        z_mean, z_log_var, next_pose = cvae(
            [tf.expand_dims(prev_pose, 0), tf.expand_dims(curr_pose, 0)]
        )
        next_pose = tf.squeeze(next_pose, 0).numpy()
        poses.append(next_pose)
        prev_pose = curr_pose
        curr_pose = next_pose

    poses = np.array(poses)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    def animate(frame):
        ax.clear()
        pose = poses[frame]

        root_x = pose[0]
        root_z = pose[1]
        root_facing = pose[2]

        x_indices = np.arange(3, 96, 3)
        y_indices = np.arange(4, 96, 3)
        z_indices = np.arange(5, 96, 3)

        x, y, z = extract_joints_xyz(pose, x_indices, y_indices, z_indices)

        cos_facing = np.cos(root_facing)
        sin_facing = np.sin(root_facing)
        rotation_matrix = np.array(
            [[cos_facing, -sin_facing], [sin_facing, cos_facing]]
        )

        rotated_xy = np.dot(rotation_matrix, np.stack([x, y]))

        world_joints = np.stack([rotated_xy[0], z, rotated_xy[1]], axis=1)
        world_joints += np.array([root_x, 0, root_z])

        ax.scatter(
            world_joints[:, 0], world_joints[:, 1], world_joints[:, 2], c="red", s=50
        )

        for bone in bones:
            if bone[0] < len(world_joints) and bone[1] < len(world_joints):
                start = world_joints[bone[0]]
                end = world_joints[bone[1]]
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]], [start[2], end[2]], "b-"
                )

        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 2)
        ax.set_zlim(-2, 2)
        ax.set_title(f"Frame {frame}")

    anim = animation.FuncAnimation(
        fig, animate, frames=len(poses), interval=200, repeat=True
    )
    plt.show()
    return anim


if __name__ == "__main__":
    anim = animate_cvae()

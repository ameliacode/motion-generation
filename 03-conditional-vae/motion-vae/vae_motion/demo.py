import os
import sys
import warnings

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "03-conditional-vae"))
sys.path.append(os.path.join(os.getcwd(), "03-conditional-vae", "motion-vae"))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import *
from mpl_toolkits.mplot3d import Axes3D

from utils.Quaternions import Quaternions

warnings.filterwarnings("ignore")

cvae = torch.load("./weights/03-conditional-vae.pt", weights_only=False)
cvae.eval()
cvae = torch.nn.DataParallel(cvae).cuda()

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
    [15, 16],
    [14, 15],
]


def process_pose(pose_data):
    if len(pose_data) >= 3:
        joints = pose_data[3:96].reshape((-1, 3))
        root_x, root_z, root_r = pose_data[0], pose_data[1], pose_data[2]
        rotation = Quaternions.from_angle_axis(-root_r, np.array([0, 1, 0]))
        transformed_joints = rotation * joints
        transformed_joints[:, 0] += root_x
        transformed_joints[:, 2] += root_z
        return transformed_joints
    return np.zeros((len(bones), 3))


def animate_plot():
    initial_pose = np.load("./data/pose0.npy")[0]
    poses = []

    condition = (
        torch.tensor(initial_pose, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to("cuda")
    )
    condition = cvae.module.normalize(condition).flatten(start_dim=1, end_dim=2)

    with torch.no_grad():
        for _ in range(OUTPUT):
            device = condition.device
            action = torch.randn(1, cvae.module.latent_size, device=device)
            vae_output = cvae.module.sample(action, condition, deterministic=True)
            vae_output = vae_output.view(
                -1, cvae.module.num_future_predictions, cvae.module.frame_size
            )
            next_frame = cvae.module.denormalize(vae_output)
            next_pose = next_frame[0, 0].cpu().numpy()
            poses.append(next_pose)
            condition = cvae.module.normalize(next_frame[:, 0:1]).flatten(
                start_dim=1, end_dim=2
            )

    processed_joints = [process_pose(p) for p in poses]
    processed_joints = np.array(processed_joints)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if len(processed_joints) > 0:
        xs, ys, zs = (
            processed_joints[:, :, 0],
            processed_joints[:, :, 1],
            processed_joints[:, :, 2],
        )
        x_range = [xs.min() - 1, xs.max() + 1]
        y_range = [ys.min() - 1, ys.max() + 1]
        z_range = [zs.min() - 1, zs.max() + 1]
    else:
        x_range = [-2, 2]
        y_range = [-1, 2]
        z_range = [-2, 2]

    def animate(frame):
        ax.clear()
        if frame < len(processed_joints):
            joints = processed_joints[frame]
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="dodgerblue", s=50)
            for start, end in bones:
                if start < len(joints) and end < len(joints):
                    ax.plot(
                        [joints[start, 0], joints[end, 0]],
                        [joints[start, 1], joints[end, 1]],
                        [joints[start, 2], joints[end, 2]],
                        "dodgerblue",
                        linewidth=8,
                        solid_capstyle="round",
                    )
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.view_init(elev=45, azim=0, roll=90)
        ax.set_title(f"Frame {frame}")

    anim = animation.FuncAnimation(
        fig, animate, frames=len(poses), interval=50, repeat=True
    )
    plt.show()
    return anim


if __name__ == "__main__":
    anim = animate_plot()
    anim.save("./03.gif", writer="pillow", fps=30)

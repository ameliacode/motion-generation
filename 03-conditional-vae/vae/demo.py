import os
import sys
import warnings

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "03-conditional-vae"))
sys.path.append(os.path.join(os.getcwd(), "03-conditional-vae", "motion-vae"))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import *
from matplotlib import animation
from model import cvae
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras

from utils.Quaternions import Quaternions

warnings.filterwarnings("ignore")
dummy_prev_pose = tf.random.normal(shape=(1, FRAME_SIZE))
dummy_curr_pose = tf.random.normal(shape=(1, FRAME_SIZE))

# Call the model once to build it
_ = cvae([dummy_prev_pose, dummy_curr_pose])
cvae.load_weights("./weights/03-conditional-vae.weights.h5")

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

# Dummy values – replace with actual stats
POSE_MEAN = np.zeros(FRAME_SIZE, dtype=np.float32)
POSE_STD = np.ones(FRAME_SIZE, dtype=np.float32)


def normalize(x):
    return (x - POSE_MEAN) / POSE_STD


def denormalize(x):
    return (x * POSE_STD) + POSE_MEAN


def process_pose(pose_data):
    joints = pose_data[3:96].reshape(-1, 3)
    root_x, root_z, root_r = pose_data[0], pose_data[1], pose_data[2]
    rotation = Quaternions.from_angle_axis(-root_r, np.array([0, 1, 0]))
    joints = rotation * joints
    joints[:, 0] += root_x
    joints[:, 2] += root_z
    return joints


def animate_keras():
    latent_size = LATENT_DIM
    num_future = NUM_FUTURE_PREDICTIONS
    frame_size = FRAME_SIZE

    pose0 = np.load("./data/pose0.npy")[0]
    cond = np.expand_dims(np.expand_dims(pose0, 0), 0)  # (1, 1, F)
    cond = normalize(cond).reshape(1, -1)

    poses = []
    for _ in range(OUTPUT):
        z = np.random.normal(size=(1, latent_size)).astype(np.float32)
        decoded = cvae.predict([z, cond])
        decoded = decoded.reshape(-1, num_future, frame_size)
        frame = denormalize(decoded)
        poses.append(frame[0, 0])
        cond = normalize(frame[:, 0:1]).reshape(1, -1)

    processed = np.array([process_pose(p) for p in poses])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        joints = processed[i]
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="dodgerblue", s=50)
        for start, end in bones:
            ax.plot(
                [joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
                [joints[start, 2], joints[end, 2]],
                "dodgerblue",
                linewidth=4,
                solid_capstyle="round",
            )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 2)
        ax.set_zlim(-2, 2)
        ax.view_init(elev=45, azim=0, roll=90)
        ax.set_title(f"Frame {i}")

    anim = animation.FuncAnimation(fig, update, frames=len(poses), interval=50)
    gif_path = "./generated_motion.gif"
    anim.save(gif_path, writer="pillow", fps=20)
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    animate_keras()

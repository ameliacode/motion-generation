import os
import sys

sys.path.append(os.path.join(os.getcwd(), "03-conditional-vae"))

from datetime import datetime

import numpy as np
import tensorflow as tf
from config import *
from keras import callbacks, optimizers
from model import *

weights_path = "./weights/03-conditional-vae.weights.h5"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"./logs/03_{timestamp}"


def normalize_data(mocap_data):
    mean = np.mean(mocap_data, axis=0)
    std = np.std(mocap_data, axis=0)
    std[std == 0] = 1.0
    normalized_data = (mocap_data - mean) / std
    return normalized_data


def create_dataset(mocap_data, batch_size, epoch, total_epochs):
    supervised_epochs = 20
    scheduled_epochs = 20

    if epoch < supervised_epochs:
        sample_prob = 1.0
    elif epoch < supervised_epochs + scheduled_epochs:
        scheduled_epoch = epoch - supervised_epochs
        sample_prob = 1.0 - (scheduled_epoch / scheduled_epochs)
    else:
        sample_prob = 0.0

    def data_generator():
        while True:
            indices = np.random.choice(
                len(mocap_data) - 2, size=batch_size, replace=True
            )
            prev_poses = mocap_data[indices]
            curr_poses = mocap_data[indices + 1]
            next_poses = mocap_data[indices + 2]

            if sample_prob < 1.0:
                use_prediction = np.random.rand(batch_size) > sample_prob
                if np.any(use_prediction):
                    _, _, predicted_curr = cvae([prev_poses, curr_poses])
                    predicted_curr = predicted_curr.numpy()

                    for i in range(batch_size):
                        if use_prediction[i]:
                            curr_poses[i] = predicted_curr[i]

            yield (prev_poses, curr_poses), next_poses

    return tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(batch_size, FRAME_SIZE), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, FRAME_SIZE), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(batch_size, FRAME_SIZE), dtype=tf.float32),
        ),
    )


num_epochs = 180
batch_size = 64
initial_lr = 1e-4
final_lr = 0.0

raw_data = np.load("./data/03_data.npz")
mocap_data = normalize_data(raw_data["data"].astype(np.float32))

steps_per_epoch = len(mocap_data) // batch_size

cvae.compile(optimizer=optimizers.Adam(learning_rate=initial_lr))

writer = tf.summary.create_file_writer(log_dir)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    if epoch < 20:
        mode = "Supervised"
        sample_prob = 1.0
    elif epoch < 40:
        mode = "Scheduled Sampling"
        scheduled_epoch = epoch - 20
        sample_prob = 1.0 - (scheduled_epoch / 20)
    else:
        mode = "Autoregressive"
        sample_prob = 0.0

    print(f"Training mode: {mode}, Sample prob: {sample_prob:.3f}")

    dataset = create_dataset(mocap_data, batch_size, epoch, num_epochs)

    lr = initial_lr + (final_lr - initial_lr) * epoch / (num_epochs - 1)
    cvae.optimizer.learning_rate.assign(lr)

    history = cvae.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)

    with writer.as_default():
        tf.summary.scalar("learning_rate", lr, step=epoch)
        tf.summary.scalar("sample_probability", sample_prob, step=epoch)
        tf.summary.scalar("training_mode", float(epoch < 20), step=epoch)

        for metric_name, metric_value in history.history.items():
            tf.summary.scalar(metric_name, metric_value[0], step=epoch)

        writer.flush()

    if epoch % 10 == 0:
        cvae.save_weights(weights_path)

cvae.save_weights(weights_path)
cvae.summary()

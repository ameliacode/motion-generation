import numpy as np
import tensorflow as tf
from config import *
from keras import optimizers
from model import *
from tqdm import tqdm

weights_path = "./weights/03-conditional-vae.weights.h5"
log_dir = "./logs"


def normalize_data(mocap_data):
    mean = np.mean(mocap_data, axis=0)
    std = np.std(mocap_data, axis=0)
    std[std == 0] = 1.0
    return (mocap_data - mean) / std


def create_batch(mocap_data, batch_size):
    indices = np.random.choice(len(mocap_data) - 1, size=batch_size, replace=True)
    prev_poses = mocap_data[indices]
    curr_poses = mocap_data[indices + 1]
    return prev_poses, curr_poses


def main():
    # Training parameters
    num_epochs = 180
    batch_size = 64
    initial_lr = 1e-4
    final_lr = 1e-7

    # Load and normalize data
    raw_data = np.load("./data/03_data.npz")
    mocap_data = normalize_data(raw_data["data"].astype(np.float32))

    # Setup model and tensorboard
    cvae.compile(optimizer=optimizers.Adam(learning_rate=initial_lr))
    writer = tf.summary.create_file_writer(log_dir)

    batches_per_epoch = len(mocap_data) // batch_size

    # Training loop with progress bar
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Update learning rate
        lr = initial_lr + (final_lr - initial_lr) * epoch / (num_epochs - 1)
        cvae.optimizer.learning_rate.assign(lr)

        epoch_loss = 0

        for batch_idx in tqdm(
            range(batches_per_epoch), desc=f"Epoch {epoch+1}", leave=False
        ):
            prev_poses, curr_poses = create_batch(mocap_data, batch_size)
            loss_dict = cvae.train_step([prev_poses, curr_poses])
            # Extract total loss (adjust key name based on your model's return)
            total_loss = loss_dict.get("loss", loss_dict.get("total_loss", 0))
            epoch_loss += total_loss

        # Log to tensorboard
        avg_loss = epoch_loss / batches_per_epoch
        with writer.as_default():
            tf.summary.scalar("total_loss", avg_loss, step=epoch)
            tf.summary.scalar("learning_rate", lr, step=epoch)
            # Log individual loss components if available
            if isinstance(loss_dict, dict):
                for loss_name, loss_value in loss_dict.items():
                    tf.summary.scalar(f"loss/{loss_name}", loss_value, step=epoch)
            writer.flush()

    # Save weights
    cvae.save_weights(weights_path)
    print(f"Training complete. Weights saved to {weights_path}")


if __name__ == "__main__":
    main()

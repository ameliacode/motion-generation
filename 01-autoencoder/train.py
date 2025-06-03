import os
import warnings
from datetime import datetime

import numpy as np
import tensorflow as tf
from config import *
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from model import autoencoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

num_epochs = EPOCHS
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"./logs/01_{timestamp}"
tensorboard_callback = TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch"
)

data = np.load("./data/01_data.npz")
X = data["clips"].astype(np.float32)  # Use float32 instead of float64 for efficiency
train_data, val_data = train_test_split(X, test_size=0.2, random_state=42)

Xmean = train_data.mean(axis=(0, 1))[np.newaxis, np.newaxis, :]
Xstd = train_data.std(axis=(0, 1))[np.newaxis, np.newaxis, :]

train_data = (train_data - Xmean) / (Xstd + 1e-8)
val_data = (val_data - Xmean) / (Xstd + 1e-8)


def loss_fn(x_real, x_pred):
    l2_loss = tf.reduce_mean(tf.square(x_real - x_pred))
    l1_loss = ALPHA * tf.reduce_mean(tf.abs(x_real - x_pred))

    total_loss = l2_loss + l1_loss

    tf.summary.scalar("l2_loss", l2_loss)
    tf.summary.scalar("l1_loss", l1_loss)

    return total_loss


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.95,
    staircase=True,
)

weights_path = "./weights/01-autoencoder.weights.h5"
if os.path.exists(weights_path):
    autoencoder.load_weights(weights_path)
    num_epochs = EPOCHS // 2

    optimizer = Adam(learning_rate=0.0005)
    autoencoder.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
else:
    optimizer = Adam(learning_rate=lr_schedule)
    autoencoder.compile(optimizer=optimizer, loss=loss_fn, metrics=["mse"])

autoencoder.summary()

history = autoencoder.fit(
    train_data,
    train_data,
    epochs=num_epochs,
    batch_size=32,
    validation_data=(val_data, val_data),
    callbacks=[tensorboard_callback],
    verbose=1,
)

autoencoder.save_weights(weights_path)

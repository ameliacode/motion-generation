import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model import autoencoder
from sklearn.model_selection import train_test_split

ALPHA = 0.01
EPOCHS = 25
BATCH_SIZE = 32

# Load and split data
data = np.load("./data/cmu_data.npz")
X = data["windows"].astype(np.float32)  # ensure float
train_data, val_data = train_test_split(X, test_size=0.1, random_state=42)


# Loss function
def loss_fn(x_real, x_pred):
    l2_loss = tf.reduce_mean(tf.square(x_real - x_pred), axis=[1, 2])
    l1_loss = ALPHA * tf.reduce_mean(tf.abs(x_pred))
    return l2_loss + l1_loss


# Compile
autoencoder.compile(optimizer="adam", loss=loss_fn)

# Train
history = autoencoder.fit(
    train_data,
    train_data,
    validation_data=(val_data, val_data),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Save weights
autoencoder.save_weights("autoencoder_weights.h5")

# Plot
plt.figure(figsize=(12, 4))
plt.plot(history.history["loss"], label="Training Loss")
if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("autoencoder_history.png")
plt.show()

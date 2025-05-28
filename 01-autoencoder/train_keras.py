import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from config import *
from model_keras import autoencoder
from sklearn.model_selection import train_test_split

data = np.load("./data/01_data.npz")
X = data["clips"].astype(np.float32)
Xmean = data["mean"].astype(np.float32)
Xstd = data["std"].astype(np.float32)
train_data, val_data = train_test_split(X, test_size=0.1, random_state=42)

train_data = (train_data - Xmean) / Xstd
val_data = (val_data - Xmean) / Xstd


def loss_fn(x_real, x_pred):
    l2_loss = tf.reduce_mean(tf.square(x_real - x_pred), axis=[1, 2])
    l1_loss = ALPHA * tf.reduce_mean(tf.abs(x_pred))
    return l2_loss + l1_loss


autoencoder.compile(optimizer="adam", loss=loss_fn)

history = autoencoder.fit(
    train_data,
    train_data,
    validation_data=(val_data, val_data),
    epochs=EPOCHS,
)

autoencoder.save_weights("01_weights.h5")

plt.figure(figsize=(12, 4))
plt.plot(history.history["loss"], label="Training Loss")
if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("01_train_loss.png")
plt.show()

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "03-conditional-vae"))

import numpy as np
import tensorflow as tf
from config import *
from keras import layers, losses, metrics, models, utils


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class CVAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        prev_pose, curr_pose = inputs
        z_mean, z_log_var, z = self.encoder([prev_pose, curr_pose])
        next_pose_prediction = self.decoder([z, prev_pose])
        return z_mean, z_log_var, next_pose_prediction

    def train_step(self, data):
        (prev_pose, curr_pose), next_pose = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, next_pose_prediction = self([prev_pose, curr_pose])

            reconstruction_loss = tf.reduce_mean(
                losses.mean_squared_error(next_pose, next_pose_prediction)
            )

            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1,
                )
            )

            total_loss = reconstruction_loss + BETA * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (prev_pose, curr_pose), next_pose = data
        z_mean, z_log_var, next_pose_prediction = self([prev_pose, curr_pose])

        reconstruction_loss = tf.reduce_mean(
            losses.mean_squared_error(next_pose, next_pose_prediction)
        )

        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                axis=1,
            )
        )

        total_loss = reconstruction_loss + BETA * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}


prev_pose = layers.Input(shape=(FRAME_SIZE,), name="previous_pose")
curr_pose = layers.Input(shape=(FRAME_SIZE,), name="current_pose")

input_poses = layers.Concatenate(name="concat_poses")([prev_pose, curr_pose])

x = layers.Dense(HIDDEN_UNITS, activation="elu", name="encoder_layer1")(input_poses)
x = layers.Dense(HIDDEN_UNITS, activation="elu", name="encoder_layer2")(x)
x = layers.Dense(HIDDEN_UNITS, activation="elu", name="encoder_layer3")(x)

z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling(name="sampling")([z_mean, z_log_var])

encoder = models.Model([prev_pose, curr_pose], [z_mean, z_log_var, z], name="encoder")

z_input = layers.Input(shape=(LATENT_DIM,), name="latent_input")
prev_pose_input = layers.Input(shape=(FRAME_SIZE,), name="previous_pose_input")

gating_input = layers.Concatenate(name="gating_concat")([z_input, prev_pose_input])
g = layers.Dense(HIDDEN_UNITS, activation="elu", name="gate_layer1")(gating_input)
g = layers.Dense(HIDDEN_UNITS, activation="elu", name="gate_layer2")(g)
g = layers.Dense(HIDDEN_UNITS, activation="elu", name="gate_layer3")(g)
gating_weights = layers.Dense(NUM_EXPERTS, activation="softmax", name="gating_weights")(
    g
)

expert_outputs = []
for i in range(NUM_EXPERTS):
    with tf.name_scope(f"expert_{i}"):
        initializer = tf.keras.initializers.RandomNormal(seed=i * 42)

        expert_input = layers.Concatenate(name=f"expert_{i}_input")(
            [z_input, prev_pose_input]
        )

        x = layers.Dense(
            HIDDEN_UNITS,
            activation="elu",
            kernel_initializer=initializer,
            name=f"expert_{i}_layer1",
        )(expert_input)

        x_with_z = layers.Concatenate(name=f"expert_{i}_concat1")([x, z_input])
        x = layers.Dense(
            HIDDEN_UNITS,
            activation="elu",
            kernel_initializer=initializer,
            name=f"expert_{i}_layer2",
        )(x_with_z)

        x_with_z = layers.Concatenate(name=f"expert_{i}_concat2")([x, z_input])
        expert_out = layers.Dense(
            FRAME_SIZE,
            activation="linear",
            kernel_initializer=initializer,
            name=f"expert_{i}_output",
        )(x_with_z)

        expert_outputs.append(expert_out)


def blend_experts(inputs):
    experts, weights = inputs
    stacked = tf.stack(experts, axis=1)
    expanded_weights = tf.expand_dims(weights, axis=-1)
    return tf.reduce_sum(stacked * expanded_weights, axis=1)


next_pose_prediction = layers.Lambda(blend_experts, name="expert_blending")(
    [expert_outputs, gating_weights]
)

decoder = models.Model([z_input, prev_pose_input], next_pose_prediction, name="decoder")

cvae = CVAE(encoder, decoder)

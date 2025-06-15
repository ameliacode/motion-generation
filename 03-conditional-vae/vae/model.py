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
        reconstruction = self.decoder([z, prev_pose])
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        prev_pose, curr_pose = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self([prev_pose, curr_pose])
            reconstruction_loss = (
                tf.reduce_mean(losses.mean_squared_error(curr_pose, reconstruction))
                * 500
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


prev_pose = layers.Input(shape=(FRAME_SIZE,), name="previous_pose")
curr_pose = layers.Input(shape=(FRAME_SIZE,), name="current_pose")
input_poses = layers.Concatenate()([prev_pose, curr_pose])
x = layers.Dense(HIDDEN_UNITS, activation="elu")(input_poses)
x = layers.Dense(HIDDEN_UNITS, activation="elu")(x)
x = layers.Dense(HIDDEN_UNITS, activation="elu")(x)
z_mean = layers.Dense(LATENT_DIM)(x)
z_log_var = layers.Dense(LATENT_DIM)(x)
z = Sampling()([z_mean, z_log_var])
encoder = models.Model([prev_pose, curr_pose], [z_mean, z_log_var, z], name="encoder")

z_input = layers.Input(shape=(LATENT_DIM,))
prev_pose_input = layers.Input(shape=(FRAME_SIZE,))

gating_input = layers.Concatenate()([z_input, prev_pose_input])
g = layers.Dense(HIDDEN_UNITS, activation="elu")(gating_input)
g = layers.Dense(HIDDEN_UNITS, activation="elu")(g)
g = layers.Dense(HIDDEN_UNITS, activation="elu")(g)
gating_weights = layers.Dense(NUM_EXPERTS, activation="softmax")(g)

expert_outputs = []
for i in range(NUM_EXPERTS):
    expert_input = layers.Concatenate()([z_input, prev_pose_input])
    x = layers.Dense(HIDDEN_UNITS, activation="elu")(expert_input)
    x_with_z = layers.Concatenate()([x, z_input])
    x = layers.Dense(HIDDEN_UNITS, activation="elu")(x_with_z)
    x_with_z = layers.Concatenate()([x, z_input])
    x = layers.Dense(HIDDEN_UNITS, activation="elu")(x_with_z)
    expert_out = layers.Dense(FRAME_SIZE, activation="linear")(x)
    expert_outputs.append(expert_out)


def blend_experts(inputs):
    experts, weights = inputs
    stacked = tf.stack(experts, axis=1)
    expanded_weights = tf.expand_dims(weights, axis=-1)
    return tf.reduce_sum(stacked * expanded_weights, axis=1)


recon_pose = layers.Lambda(blend_experts)([expert_outputs, gating_weights])
decoder = models.Model([z_input, prev_pose_input], recon_pose, name="decoder")

cvae = CVAE(encoder, decoder)

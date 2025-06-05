import tensorflow as tf
from config import *
from keras import backend as K
from keras import layers, losses, metrics, models, utils


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
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
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                500 * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
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


## ENCODER ##
prev_pose = layers.Input(shape=(), name="previous_pose")
curr_pose = layers.Input(shape=(), name="current_pose")
input_poses = layers.Concatenate(name="input_poses")([prev_pose, curr_pose])
x = layers.Dense(HIDDEN_UNITS, activation="elu")(input_poses)
x = layers.Dense(HIDDEN_UNITS, activation="elu")(x)
x = layers.Dense(HIDDEN_UNITS, activation="elu")(x)

z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model([prev_pose, curr_pose], [z_mean, z_log_var, z], name="encoder")
encoder.summary()

## MoE DECODER ##
z_input = layers.Input(shape=(LATENT_DIM,), name="latent_input")

# Gating Network
gating_input = layers.Concatenate(name="gating_input")([z_input, prev_pose])
g = layers.Dense(HIDDEN_UNITS, activation="elu")(gating_input)
g = layers.Dense(HIDDEN_UNITS, activation="elu")(g)
g = layers.Dense(HIDDEN_UNITS, activation="elu")(g)
gating_weights = layers.Dense(NUM_EXPERTS, activation="softmax")(g)

# Expert Networks
expert_outputs = []
for i in range(NUM_EXPERTS):
    expert_input = layers.Concatenate()([z_input, prev_pose])
    x = layers.Dense(HIDDEN_UNITS, activation="elu")(expert_input)

    x_with_z = layers.Concatenate()([x, z_input])
    x = layers.Dense(HIDDEN_UNITS, activation="elu")(x_with_z)

    x_with_z = layers.Concatenate()([x, z_input])
    x = layers.Dense(HIDDEN_UNITS, activation="elu")(x_with_z)

    expert_out = layers.Dense(POSE_DIM, activation="linear")(x)
    expert_outputs.append(expert_out)


# Blend experts
def blend_experts(inputs):
    experts, weights = inputs
    stacked = tf.stack(experts, axis=1)
    expanded_weights = tf.expand_dims(weights, axis=-1)
    return tf.reduce_sum(stacked * expanded_weights, axis=1)


recon_pose = layers.Lambda(blend_experts)([expert_outputs, gating_weights])

decoder = models.Model([z_input, prev_pose], recon_pose, name="moe_decoder")
decoder.summary()

cvae = CVAE(encoder, decoder)

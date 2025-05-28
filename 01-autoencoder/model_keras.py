from config import *
from keras import layers, models

encoder_input = layers.Input(shape=(WINDOW_SIZE, CHANNEL_SIZE))
x = layers.Conv1D(64, kernel_size=FILTER_SIZE, strides=1, padding="same")(encoder_input)
x = layers.MaxPooling1D(padding="same")(x)
x = layers.Activation("tanh")(x)
x = layers.Conv1D(128, kernel_size=FILTER_SIZE, strides=1, padding="same")(x)
x = layers.MaxPooling1D(padding="same")(x)
x = layers.Activation("tanh")(x)
x = layers.Conv1D(256, kernel_size=FILTER_SIZE, strides=1, padding="same")(x)
x = layers.MaxPooling1D(padding="same")(x)
encoder_output = layers.Activation("tanh")(x)

encoder = models.Model(encoder_input, encoder_output)
# encoder.summary()

decoder_input = layers.Input(shape=(NUM_JOINTS, 256))

x = layers.Activation("tanh")(decoder_input)
x = layers.UpSampling1D(size=2)(x)
x = layers.Conv1D(
    128, kernel_size=FILTER_SIZE, strides=1, padding="same", activation="relu"
)(x)
x = layers.Activation("tanh")(x)
x = layers.UpSampling1D(size=2)(x)
x = layers.Conv1D(
    64, kernel_size=FILTER_SIZE, strides=1, padding="same", activation="relu"
)(x)
x = layers.Activation("tanh")(x)
x = layers.UpSampling1D(size=2)(x)
decoder_output = layers.Conv1D(
    63, kernel_size=FILTER_SIZE, strides=1, padding="same", activation="tanh"
)(x)

decoder = models.Model(decoder_input, decoder_output)
# decoder.summary()

autoencoder = models.Model(encoder_input, decoder(encoder_output))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Conv1D layers
        self.conv1 = nn.Conv1d(
            CHANNEL_SIZE, 64, kernel_size=FILTER_SIZE, stride=1, padding="same"
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            64, 128, kernel_size=FILTER_SIZE, stride=1, padding="same"
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(
            128, 256, kernel_size=FILTER_SIZE, stride=1, padding="same"
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.tanh(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.tanh(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = torch.tanh(x)

        x = x.transpose(1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv1d(
            256, 128, kernel_size=FILTER_SIZE, stride=1, padding="same"
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv1d(
            128, 64, kernel_size=FILTER_SIZE, stride=1, padding="same"
        )

        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3 = nn.Conv1d(
            64, 63, kernel_size=FILTER_SIZE, stride=1, padding="same"
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = torch.tanh(x)
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
        x = torch.tanh(x)

        x = self.upsample2(x)
        x = F.relu(self.conv2(x))
        x = torch.tanh(x)

        x = self.upsample3(x)
        x = torch.tanh(self.conv3(x))

        x = x.transpose(1, 2)
        return x


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

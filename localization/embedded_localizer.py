from numpy import pad
import torch
from torch import nn
from general_layers.conv import Conv, ResidualConv


class EmbeddedLocalizer(nn.Module):
    """
    Experimental localization module for predicting camera pose from latent space vectors.
    """
    def __init__(self, in_channels=3, latent_vector_features=2048, channels=[32, 64, 128, 128, 128, 256, 256, 128, 512]):
            super(EmbeddedLocalizer, self).__init__()

            self.flattened_shape = 1536

            self.net = nn.Sequential(
                Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=[7, 7], padding=[3, 3]),
                
                Conv(in_channels=in_channels, out_channels=channels[0], kernel_size=3, stride=2, padding=1),
                Conv(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=2, padding=1),
                Conv(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=2, padding=1),
                Conv(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=2, padding=1),
                Conv(in_channels=channels[3], out_channels=channels[4], kernel_size=3, stride=2, padding=1),
                Conv(in_channels=channels[4], out_channels=channels[5], kernel_size=3, stride=2, padding=1),
                Conv(in_channels=channels[5], out_channels=channels[6], kernel_size=3, stride=2, padding=1),
                Conv(in_channels=channels[6], out_channels=channels[7], kernel_size=3, stride=2, padding=1),
                Conv(in_channels=channels[7], out_channels=channels[8], kernel_size=3, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(in_features=self.flattened_shape, out_features=latent_vector_features)
            )

    def forward(self, input):
        return self.net(input)

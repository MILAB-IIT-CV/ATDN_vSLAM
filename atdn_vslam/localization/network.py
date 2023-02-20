import torch
from torch import nn

from ..layers.conv import Conv, ResidualConv, DUC, TransposedConv
from ..utils.normalizations import get_rgb_norm
from ..utils.helpers import log, ShapeLogLayer


class MappingVAE(nn.Module):
    def __init__(
        self,
        variational=False
    ):
        super(MappingVAE, self).__init__()

        in_channels = 3
        out_channels = in_channels

        channels=[16, 16, 32, 64, 128, 128, 256]

        self.normalization = get_rgb_norm()
        self.var = variational
        self.flattened_shape = 1536

        DownScaler = ResidualConv
        UpScaler = DUC
        activation = nn.Mish

        self.encoder = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=[7, 7], padding=[3, 3], activation=activation),            
            DownScaler(in_channels=in_channels, out_channels=channels[0], stride=2, activation=activation),
            DownScaler(in_channels=channels[0], out_channels=channels[1], stride=2, activation=activation),
            DownScaler(in_channels=channels[1], out_channels=channels[2], stride=2, activation=activation),
            DownScaler(in_channels=channels[2], out_channels=channels[3], stride=2, activation=activation),
            DownScaler(in_channels=channels[3], out_channels=channels[4], stride=2, activation=activation),
            DownScaler(in_channels=channels[4], out_channels=channels[5], stride=2, activation=activation),
            #DownScaler(in_channels=channels[5], out_channels=channels[6], stride=3, activation=activation),
            #nn.Flatten(),
            #nn.Linear(in_features=3584, out_features=1024),
            #activation(),
            #nn.Dropout(0.2)
        )

        #self.sigma_lin = nn.Conv2d(in_channels=channels[4], out_channels=channels[4], kernel_size=3, stride=1)
        self.mean_lin  = nn.Conv2d(in_channels=channels[5], out_channels=channels[5], kernel_size=1, stride=1)

        self.decoder = nn.Sequential(
            TransposedConv(in_channels=channels[5], out_channels=channels[4], kernel_size=3, stride=2, padding=1),
            TransposedConv(in_channels=channels[4], out_channels=channels[3], kernel_size=3, stride=2, padding=1),
            TransposedConv(in_channels=channels[3], out_channels=channels[2], kernel_size=3, stride=2, padding=1),
            TransposedConv(in_channels=channels[2], out_channels=channels[1], kernel_size=3, stride=2, padding=1),
            TransposedConv(in_channels=channels[1], out_channels=channels[0], kernel_size=3, stride=2, padding=1),
            TransposedConv(in_channels=channels[0], out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=3, padding=1)
        )

    def forward(self, image):
        normalized = self.normalization(image)

        encoded = self.encoder(normalized)
        #log("Encoded shape: ", encoded.shape)
        
        if self.var:
            mu = self.mean_lin(encoded)
            logvar = self.sigma_lin(encoded)
            sigma = torch.exp(torch.tensor(0.5)*logvar)
            eps = torch.randn_like(sigma)
            latent_vector = mu + sigma*eps
        else:
            latent_vector = self.mean_lin(encoded)
            mu = latent_vector
            logvar = None


        decoded = self.decoder(latent_vector)
        
        return mu, logvar, latent_vector, decoded
        #return decoded

    def get_code(self, image):
        code = self.encoder(image)
        mu = self.mean_lin(code)
        normed = self.norm(mu)
        return mu, code, normed

    def generate_from_code(self, code):
        return self.decoder(code)

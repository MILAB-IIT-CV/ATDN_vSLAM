import torch
from torch import nn
from general_layers import conv


class DepthPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            conv.Conv(in_channels=8, out_channels=8, kernel_size=7, padding=3),
            conv.ResidualConv(in_channels=8, out_channels=16),
            conv.ResidualConv(in_channels=16, out_channels=32),
            conv.ResidualConv(in_channels=32, out_channels=64),
            conv.ResidualConv(in_channels=64, out_channels=64),
            conv.ResidualConv(in_channels=64, out_channels=64),
            conv.ResidualConv(in_channels=64, out_channels=64),
            conv.ResidualConv(in_channels=64, out_channels=32),
            conv.ResidualConv(in_channels=32, out_channels=16),
            conv.ResidualConv(in_channels=16, out_channels=8),
            conv.ResidualConv(in_channels=8, out_channels=4),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        )


    def forward(self, flow, rgb1, rgb2):
        input = torch.cat([flow, rgb1, rgb2], dim=1)
        depths = self.net(input)
        return depths[:, 0, :, :], depths[:, 1, :, :]
import torch
from torch import nn
from layers.linear import Linear


class Symmetric(nn.Module):
    def __init__(self, channels, activation=nn.Mish) -> None:
        super().__init__()

        self.pos = activation()
        self.neg = activation()
        self.conv = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1)


    def forward(self, input):

        x_pos = self.pos(input)
        x_neg = (-1.0)*self.neg((-1.0)*input)

        x = torch.cat([x_pos, x_neg], dim=1)
        x = self.conv(x)

        return x


class AutoCorr(nn.Module):
    def __init__(self, in_features, map_size):
        super().__init__()
        
        self.linear_mapping = nn.Linear(in_features=in_features, out_features=map_size)
        self.flatten = nn.Flatten()

    def forward(self, input):
        vec = self.linear_mapping(input)

        v1 = vec.unsqueeze(2)
        v2 = vec.unsqueeze(1)

        corr = torch.matmul(v1, v2)
        
        corr = self.flatten(corr)

        return corr


class Spatial(nn.Module):
    def __init__(
        self, 
        in_height, 
        in_width, 
        hidden, 
        channels
    ) -> None:
        super().__init__()
        

        self.in_hlin = Linear(in_features=in_height, out_features=hidden)
        self.in_wlin = Linear(in_features=in_width, out_features=hidden)

        self.out_norm = nn.BatchNorm2d(num_features=channels)
        

    def forward(self, input):
        # TODO layernorm
        hx = self.in_hlin(input.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        wx = self.in_wlin(input)

        x = wx @ hx

        x = self.out_norm(x)

        return x


class ResidualSpatial(nn.Module):
    def __init__(
        self,
        in_height, 
        in_width, 
        hidden, 
        channels,
        stride
    ) -> None:
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.Mish(),
            nn.BatchNorm2d(num_features=channels)
        )
        self.spatial = Spatial(in_height, in_width, hidden, channels)
        self.spatial_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=1)

        #self.skip_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=1)
        self.skip_pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        self.out_activation = nn.Mish()
        self.out_bn = nn.BatchNorm2d(num_features=channels)


    def forward(self, input):
        skip = self.skip_pool(input)
        
        x = self.in_conv(input)
        x = self.spatial(x)
        x = self.spatial_conv(x)

        x = x + skip

        x = self.out_bn(self.out_activation(x))

        return x

class SAct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input):
        return torch.sign(input)*torch.sqrt(input.abs())
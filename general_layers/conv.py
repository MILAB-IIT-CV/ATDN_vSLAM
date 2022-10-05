from torchvision.transforms import Resize
from torch import nn
import torch

from utils.helpers import log



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=[1, 1], dilation=[1, 1], padding=[0, 0], activation=nn.Mish) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding)
        self.activation = activation(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        return self.bn(self.activation(self.conv(input)))


class ResidualConv(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, activation=nn.Mish):
    super(ResidualConv, self).__init__()

    self.conv = nn.Sequential(
      Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, activation=activation),
      Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, activation=activation)
    )

    self.skip_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    self.out_block = nn.Sequential(
      nn.Mish(),
      nn.BatchNorm2d(num_features=out_channels)
    )

  def forward(self, input):
    x = self.conv(input)  
    skip = self.skip_layer(input)
    x = x + skip
    
    return self.out_block(x)


class TransposedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), output_padding=(0, 0)) -> None:
        super(TransposedConv, self).__init__()

        self.conv = nn.Sequential(
                nn.ConvTranspose2d( in_channels=in_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size, 
                                    stride=stride,
                                    padding=padding,
                                    output_padding=output_padding),
                nn.PReLU(),
                nn.BatchNorm2d(num_features=out_channels)
            )


    def forward(self, input):
        return self.conv(input)


class InterleaveUpscaling(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.PReLU):
    super(InterleaveUpscaling, self).__init__()
    
    self.out_channels = out_channels
    
    #self.conv1 = nn.Sequential(
    #    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    #    nn.PReLU(),
    #    nn.BatchNorm2d(num_features=out_channels)
    #)

    #self.conv2 = nn.Sequential(
    #    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    #    nn.PReLU(),
    #    nn.BatchNorm2d(num_features=out_channels)
    #)
    
    #self.conv3 = nn.Sequential(
    #    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    #    nn.PReLU(),
    #    nn.BatchNorm2d(num_features=out_channels)
    #)

    #self.conv4 = nn.Sequential(
    #    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    #    nn.PReLU(),
    #    nn.BatchNorm2d(num_features=out_channels)
    #)

    self.conv = Conv(in_channels=in_channels, out_channels=4*out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)

    self.reshuffle = nn.PixelShuffle(2)

  def forward(self, input):
    #x1 = self.conv1(input)
    #x2 = self.conv2(input)
    #x3 = self.conv3(input)
    #x4 = self.conv4(input)
    
    #x = torch.cat([x1, x2, x3, x4], dim=1)
    x = self.conv(input)

    x = self.reshuffle(x)
    
    return x



class ConnectedUpscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation) -> None:
        super().__init__()

        self.pre_conv = Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)
        #self.post_conv = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.post_conv = Conv(in_channels=in_channels*2, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)


    def forward(self, input, shortcut):
        resize = Resize(shortcut.shape[-2:])
        x = resize(input)
        x = self.pre_conv(x) # TODO try pre_conv on shortcut
        #x = x + shortcut
        x = torch.cat([x, shortcut], dim=-3)
        x = self.post_conv(x)

        return x

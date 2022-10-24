from torchvision.transforms import Resize
from torch import nn
import torch

from utils.helpers import log


class Conv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride = 1, 
        dilation = 1, 
        padding = 0, 
        activation = nn.Mish, 
        bias=True,
        init=False
    ) -> None:
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              dilation=dilation, 
                              padding=padding,
                              bias=bias)

        self.activation = activation(inplace=True)

        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        return self.bn(self.activation(self.conv(input)))


class ResidualConv(nn.Module):
  def __init__(
      self, 
      in_channels, 
      out_channels, 
      stride = 1, 
      activation = nn.Mish,
      init = False
  ):
    super(ResidualConv, self).__init__()

    self.conv = nn.Sequential(
      Conv(in_channels=in_channels, 
           out_channels=in_channels, 
           kernel_size=3, 
           stride=1, 
           padding=1, 
           activation=activation, 
           bias=False),

      Conv(in_channels=in_channels, 
           out_channels=out_channels, 
           kernel_size=3, 
           stride=stride, 
           padding=1, 
           activation=activation, 
           bias=False)
    )

    self.skip_layer = nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=1, 
                                stride=stride, 
                                bias=False)


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
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride = 1, 
        padding = 0, 
        output_padding = 0
    ) -> None:
        super(TransposedConv, self).__init__()

        self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, 
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


class DUC(nn.Module):
  def __init__(
      self, 
      in_channels, 
      out_channels, 
      kernel_size, 
      stride, 
      padding, 
      activation=nn.Mish
  ):
    super(DUC, self).__init__()
    
    self.out_channels = out_channels
    
    self.conv = Conv(in_channels=in_channels, 
                     out_channels=4*out_channels, 
                     kernel_size=kernel_size, 
                     stride=stride, 
                     padding=padding, 
                     activation=activation)

    self.reshuffle = nn.PixelShuffle(2)

  def forward(self, input):

    x = self.conv(input)

    x = self.reshuffle(x)
    
    return x


class ConnectedDUC(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels,
    ) -> None:
       super(ConnectedDUC, self).__init__()
       
       self.duc = DUC(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
       self.connector = Conv(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, padding=1)


    def forward(self, direct, skip):
        x = torch.cat([direct, skip], dim=1)
        x = self.connector(x)
        x = self.duc(x)

        return x


class ConnectedUpscale(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding, 
        activation
    ) -> None:
        super().__init__()

        self.pre_conv = Conv(in_channels=in_channels, 
                             out_channels=in_channels, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             padding=padding, 
                             activation=activation)

        self.post_conv = Conv(in_channels=in_channels*2, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              activation=activation)


    def forward(self, input, shortcut):
        resize = Resize(shortcut.shape[-2:])
        x = resize(input)
        x = self.pre_conv(x) # TODO try pre_conv on shortcut
        #x = x + shortcut
        x = torch.cat([x, shortcut], dim=-3)
        x = self.post_conv(x)

        return x

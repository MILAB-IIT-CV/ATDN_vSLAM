from torchvision.transforms import Resize
from torch import nn
import torch

from utils.helpers import log


class Conv(nn.Module):
    """
    Basic Conv-Activation-Batchnorm2d block. Currently Mish is used as the activation function.
    """
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
    """
    Residual convolution block
    """
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
    """
    Transposed Conv block. Consists of TransposedConv2d-Activation-BatchNorm2d
    """
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
    """
    Dense upscaling convolution block. Consists of Conv block and pixelshuffle
    """
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
    """
    Connected Dense Upscaling convolution block for Unet structures. Concatenates the two (direct and skip) inputs, performs one Conv block and one DUC block on it.
    """
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
    """
    Connected upscaling block. Performs Conv block on input, concatenates with skip and performs another Conv on them.
    """
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


    def forward(self, direct, skip):
        resize = Resize(skip.shape[-2:])
        x = resize(direct)
        x = self.pre_conv(x) # TODO try pre_conv on shortcut
        #x = x + shortcut
        x = torch.cat([x, skip], dim=-3)
        x = self.post_conv(x)

        return x

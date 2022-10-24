from torchvision.transforms.functional import crop
from torchvision.transforms import Resize
from torch import nn
from general_layers import conv
from utils.helpers import log

class DepthPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        #self.net = nn.Sequential(
        #    conv.Conv(in_channels=8, out_channels=16, kernel_size=7, padding=3),
        #    conv.Conv(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        #    conv.Conv(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        #    conv.Conv(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        #    conv.Conv(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        #    conv.Conv(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        #    conv.Conv(in_channels=16, out_channels=8, kernel_size=3, padding=1),
        #    nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, padding=1)
        #)

        self.stem = conv.Conv(in_channels=8, out_channels=8, kernel_size=7, stride=2, padding=3)

        self.E1 = conv.ResidualConv(in_channels=8, out_channels=16, stride=2)
        self.E2 = conv.ResidualConv(in_channels=16, out_channels=32, stride=2)
        self.E3 = conv.ResidualConv(in_channels=32, out_channels=64, stride=2)
        self.E4 = conv.ResidualConv(in_channels=64, out_channels=128, stride=2)
        self.E5 = conv.ResidualConv(in_channels=128, out_channels=128, stride=2)
        
        self.bottleneck = nn.Sequential(
            conv.ResidualConv(in_channels=128, out_channels=256),
            conv.ResidualConv(in_channels=256, out_channels=256),
            conv.ResidualConv(in_channels=256, out_channels=128),
        )

        self.De1 = conv.ConnectedDUC(in_channels=128, out_channels=128)
        self.De2 = conv.ConnectedDUC(in_channels=128, out_channels=64)
        self.De3 = conv.ConnectedDUC(in_channels=64, out_channels=32)
        self.De4 = conv.ConnectedDUC(in_channels=32, out_channels=16)
        self.De5 = conv.ConnectedDUC(in_channels=16, out_channels=2)

        self.resize = Resize((376, 1241))


    def forward(self, input):
        #log(input.shape)
        pre = self.stem(input)
        #log(pre.shape)
        down1 = self.E1(pre)
        #log(down1.shape)
        down2 = self.E2(down1)
        #log(down2.shape)
        down3 = self.E3(down2)
        #log(down3.shape)
        down4 = self.E4(down3)
        #log(down4.shape)
        down5 = self.E5(down4)
        #log(down5.shape)

        bottom = self.bottleneck(down5)

        shape = bottom.shape
        up5 = self.De1(bottom, crop(down5, 0, 0, shape[2], shape[3]))
        #log(up5.shape)
        shape = up5.shape
        up4 = self.De2(up5, crop(down4, 0, 0, shape[2], shape[3]))
        #log(up4.shape)
        shape = up4.shape
        up3 = self.De3(up4, crop(down3, 0, 0, shape[2], shape[3]))
        #log(up3.shape)
        shape = up3.shape
        up2 = self.De4(up3, crop(down2, 0, 0, shape[2], shape[3]))
        #log(up2.shape)
        shape = up2.shape
        up1 = self.De5(up2, crop(down1, 0, 0, shape[2], shape[3]))
        #log(up1.shape)

        depths = self.resize(up1)

        return depths[:, 0, :, :], depths[:, 1, :, :]


        #depths = self.net(input)
        #return depths[:, 0, :, :], depths[:, 1, :, :]
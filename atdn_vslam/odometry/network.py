import torch
from torch import nn
from torchvision.transforms import Normalize
from ..layers.conv import Conv, ResidualConv
from ..layers.linear import Linear

from ..utils.normalizations import get_flow_norm
from ..utils.helpers import log, ShapeLogLayer


class ATDNVO(nn.Module):
    __doc__="""
    Odometry network of the SLAM system.
    
    :param batch_size: Number of data batched as input and output
    :param in_channels: Number of channels of input data. Used for easy training experiments.
    :param device: Destination device for the model.
    
    """
    def __init__(
        self, 
        batch_size : int = 1, 
        in_channels : int = 2,
        compressor = True,
        use_dropout = False,
        use_layernorm = False,
    ):
        super().__init__()

        # ------------------------------
        # Storing init object parameters
        # ------------------------------
        
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.device = "cpu"

        # ----------------------
        # Implicit normalization
        # ----------------------
        
        self.normalize_flow = get_flow_norm()
        self.polar_norm = nn.BatchNorm2d(num_features=2)

        # ----------------------------------------------------
        # Feature extractor encoder module for the LSTM module
        # ----------------------------------------------------

        activation = nn.Mish

        # TODO uppercase suffixes
        self.suffix = ""
        if compressor:
            self.suffix += "c"
        if use_layernorm:
            self.suffix += "l"
        if use_dropout:
            self.suffix += "d"
        if self.suffix != "":
            self.suffix = "_" + self.suffix

        if compressor:
            self.encoder_CNN = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, groups=in_channels), # TODO omit this for it can cause overfitting
                Conv(in_channels=self.in_channels, out_channels=16, kernel_size=7, stride=2, padding=3, activation=activation, bias=True),
                ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
                ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
                ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
                ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
                Conv(in_channels=16, out_channels=16, kernel_size=3, stride=3, padding=0, activation=activation),
                nn.Flatten(),
                Linear(in_features=832, out_features=512, activation=activation, dropout=use_dropout, norm=use_layernorm)
            )        
        else:
            self.encoder_CNN = nn.Sequential(
                #nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, groups=in_channels),
                #Conv(in_channels=self.in_channels, out_channels=16, kernel_size=7, stride=2, padding=3, activation=activation, bias=True),
                ResidualConv(in_channels=self.in_channels, out_channels=4, stride=2, activation=activation),
                ResidualConv(in_channels=4, out_channels=8, stride=2, activation=activation),
                ResidualConv(in_channels=8, out_channels=16, stride=2, activation=activation),
                ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
                ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
                ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
                nn.Flatten(),
                Linear(in_features=1920, out_features=512, activation=activation, dropout=use_dropout, norm=use_layernorm)
            )        
        
        # --------------------------------------------------
        # Blocks of the LSTM (Long Short Term Memory) module
        # --------------------------------------------------
        self.lstm_out_size = 512
        self.lstm1 = nn.LSTMCell(input_size=512,
                                 hidden_size=self.lstm_out_size)

        self.lstm1_h = torch.zeros(self.batch_size, self.lstm_out_size)
        self.lstm1_c = torch.zeros(self.batch_size, self.lstm_out_size)

        self.lstm_linear = Linear(in_features=512, out_features=512, activation=activation, dropout=use_dropout, norm=use_layernorm)

        self.lstm2 = nn.LSTMCell(input_size=self.lstm_out_size,
                                 hidden_size=self.lstm_out_size)

        self.lstm2_h = torch.zeros(self.batch_size, self.lstm_out_size)
        self.lstm2_c = torch.zeros(self.batch_size, self.lstm_out_size)

        # --------------------------------------
        # MLP heads for rotation and translation
        # --------------------------------------
        self.translation_regressor = nn.Sequential(
                Linear(in_features=self.lstm_out_size, out_features=128, activation=activation, dropout=use_dropout, norm=use_layernorm),
                Linear(in_features=128, out_features=64, activation=activation, dropout=use_dropout, norm=use_layernorm),
                nn.Linear(in_features=64, out_features=3, bias=False)
        )

        self.rotation_regressor = nn.Sequential(
                Linear(in_features=self.lstm_out_size, out_features=128, activation=activation, dropout=use_dropout, norm=use_layernorm),
                Linear(in_features=128, out_features=64, activation=activation, dropout=use_dropout, norm=use_layernorm),
                nn.Linear(in_features=64, out_features=3, bias=False)
        )


    def forward(self, flows : torch.Tensor):
        """
        The "call" function of the odometry estimator.

        :param flows: Input optical flow values of shape (Batch, 2, Height, Width)
        :type flows: torch.Tensor
        :return: The estimated euler angle rotation vector and estimated translation vector
        :rtype: torch.Tensor, torch.Tensor
        """
        normalized_flows = self.normalize_flow(flows)

        # Extracted features
        features = self.encoder_CNN(normalized_flows)

        # Long Short Term Memory
        [self.lstm1_h, self.lstm1_c] = self.lstm1(features, [self.lstm1_h, self.lstm1_c])
        lstm2_input = self.lstm_linear(self.lstm1_h)
        [self.lstm2_h, self.lstm2_c] = self.lstm2(lstm2_input, [self.lstm2_h, self.lstm2_c])
        lstm_out = self.lstm2_h

        # Odometry module translation branch
        rotations = self.rotation_regressor(lstm_out)
        translations = self.translation_regressor(lstm_out)

        return rotations, translations


    def reset_lstm(self):
        self.lstm1_h = torch.zeros(self.batch_size, self.lstm_out_size, device=self.device)
        self.lstm1_c = torch.zeros(self.batch_size, self.lstm_out_size, device=self.device)
        self.lstm2_h = torch.zeros(self.batch_size, self.lstm_out_size, device=self.device)
        self.lstm2_c = torch.zeros(self.batch_size, self.lstm_out_size, device=self.device)


    def to(self, device):
        super().to(device)
        
        self.device = device
        self.reset_lstm()

        return self

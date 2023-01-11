import torch
from torch import nn
from torchvision.transforms import Normalize
from layers.conv import Conv, ResidualConv, SimConv, SimResidual
from layers.linear import Linear, SimLin
from layers.experimental import AutoCorr
from utils.helpers import log, ShapeLogLayer


class LSCVO(nn.Module):
    __doc__="""
    Odometry network of the SLAM system.
    
    :param batch_size: Number of data batched as input and output
    :param in_channels: Number of channels of input data. Used for easy training experiments.
    :param device: Destination device for the model.
    
    """
    def __init__(
        self, 
        batch_size : int = 1, 
        in_channels : int = 2
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
        #self.normalize_flow = Normalize(mean=[0.0, 0.0], std=[41.2430, 41.1322])
        self.normalize_flow = Normalize(mean=[0.0, 0.0], std=[58.1837, 17.7647])
        #self.normalize_depth = Normalize(mean=[0.7178], std=[0.7966])

        # ----------------------------------------------------
        # Feature extractor encoder module for the LSTM module
        # ----------------------------------------------------

        activation = nn.Mish
        #activation = nn.PReLU
        use_dropout = True # TODO change names for atdn and atdnd in checkpoints
        use_layernorm = False
        symmetric = False
        # TODO Try crosscorrelation matmul
        self.suffix = ""

        self.encoder_CNN = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, groups=in_channels),
            Conv(in_channels=self.in_channels, out_channels=16, kernel_size=7, stride=2, padding=3, activation=activation, bias=True),
            ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
            ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
            ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
            ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation),
            ResidualConv(in_channels=16, out_channels=16, stride=2, activation=activation), # Not needed for comp version
            #Conv(in_channels=16, out_channels=16, kernel_size=3, stride=3, padding=0, activation=activation), # Not needed with ATDN version
            nn.Flatten()
        )        
        self.pre_lstm_lin = Linear(in_features=1920, out_features=512, activation=activation, dropout=use_dropout, norm=use_layernorm) # 1920 for ATDN 832 with comp
        
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

        # TODO try with simple nn.Linear
        self.shortcut = Linear(in_features=1920, out_features=512)
        self.merger = nn.Sequential(
            Linear(in_features=1024, out_features=512),
            nn.LayerNorm(normalized_shape=512)
        )

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
        # ------------------
        # Extracted features
        # ------------------
        cnn_features = self.encoder_CNN(normalized_flows)
        features = self.pre_lstm_lin(cnn_features)
        #log("Encoder out: ", features.shape)

        # ----------------------
        # Long Short Term Memory
        # ----------------------
        [self.lstm1_h, self.lstm1_c] = self.lstm1(features, [self.lstm1_h, self.lstm1_c])
        lstm2_input = self.lstm_linear(self.lstm1_h)
        [self.lstm2_h, self.lstm2_c] = self.lstm2(lstm2_input, [self.lstm2_h, self.lstm2_c])
        lstm_out = self.lstm2_h

        skip = self.shortcut(cnn_features)
        lstm_out = self.merger(torch.cat([lstm_out, skip], dim=1))
        # ----------------------------------
        # Odometry module translation branch
        # ----------------------------------
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

import torch
from torch import nn
from torchvision.transforms import Normalize
from layers.conv import Conv, ResidualConv, SimConv, SimResidual
from layers.linear import Linear, SimLin
from layers.experimental import AutoCorr
from utils.helpers import log, ShapeLogLayer


class SymVO(nn.Module):
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
        use_dropout = True
        symmetric = False
        # TODO Try crosscorrelation matmul
        self.suffix = ""

        self.encoder_CNN = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, groups=in_channels),
            SimConv(in_channels=self.in_channels, out_channels=4, kernel_size=7, stride=2, padding=3, activation=activation),
            SimResidual(in_channels=4, out_channels=4, stride=2, activation=activation),
            SimResidual(in_channels=4, out_channels=8, stride=2, activation=activation),
            SimResidual(in_channels=8, out_channels=8, stride=2, activation=activation),
            SimResidual(in_channels=8, out_channels=16, stride=2, activation=activation),
            SimResidual(in_channels=16, out_channels=16, stride=2, activation=activation),
            nn.Flatten(),
            SimLin(in_features=1920, out_features=512, dropout=True, activation=activation),
        )
        
        # --------------------------------------------------
        # Blocks of the LSTM (Long Short Term Memory) module
        # --------------------------------------------------
        self.lstm_out_size = 512
        self.lstm1 = nn.LSTMCell(input_size=512,
                                 hidden_size=self.lstm_out_size)

        self.lstm1_h = torch.zeros(self.batch_size, self.lstm_out_size)
        self.lstm1_c = torch.zeros(self.batch_size, self.lstm_out_size)

        self.lstm_linear = SimLin(in_features=512, out_features=512, activation=activation, dropout=use_dropout)

        self.lstm2 = nn.LSTMCell(input_size=self.lstm_out_size,
                                 hidden_size=self.lstm_out_size)

        self.lstm2_h = torch.zeros(self.batch_size, self.lstm_out_size)
        self.lstm2_c = torch.zeros(self.batch_size, self.lstm_out_size)

        #self.rot_mapper1 = SimLin(in_features=self.lstm_out_size, out_features=256)
        #self.rotation_corr = AutoCorr(in_features=256, map_size=16)

        #self.tr_mapper1 = SimLin(in_features=self.lstm_out_size, out_features=256)
        #self.translation_corr = AutoCorr(in_features=256, map_size=16)

        # --------------------------------------
        # MLP heads for rotation and translation
        # --------------------------------------
        self.translation_regressor = nn.Sequential(
                SimLin(in_features=self.lstm_out_size, out_features=128, activation=activation),
                SimLin(in_features=128, out_features=64, activation=activation),
                nn.Linear(in_features=64, out_features=3, bias=False)
        )

        self.rotation_regressor = nn.Sequential(
                SimLin(in_features=self.lstm_out_size, out_features=128, activation=activation),
                SimLin(in_features=128, out_features=64, activation=activation),
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
        features = self.encoder_CNN(normalized_flows)
        #log("Encoder out: ", features.shape)

        # ----------------------
        # Long Short Term Memory
        # ----------------------
        [self.lstm1_h, self.lstm1_c] = self.lstm1(features, [self.lstm1_h, self.lstm1_c])
        lstm2_input = self.lstm_linear(self.lstm1_h)
        [self.lstm2_h, self.lstm2_c] = self.lstm2(lstm2_input, [self.lstm2_h, self.lstm2_c])
        lstm_out = self.lstm2_h

        # Autocorr
        #rot_map = self.rot_mapper1(lstm_out)
        #rot_autocor = self.rotation_corr(rot_map)
        #rotation_input = torch.cat([rot_map, rot_autocor], dim=1)

        #tr_map = self.tr_mapper1(lstm_out)
        #tr_autocor = self.translation_corr(tr_map)
        #translation_input = torch.cat([tr_map, tr_autocor], dim=1)

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
    

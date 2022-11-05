import torch
from torch import nn
from torchvision.transforms import Normalize
from general_layers.conv import Conv, ResidualConv

from utils.helpers import log


class CLVO(nn.Module):
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
        device : str = "cuda:0"
    ):
        super(CLVO, self).__init__()

        # ------------------------------
        # Storing init object parameters
        # ------------------------------
        
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.device = device

        self.activation = nn.PReLU
        self.MLP_activation = nn.PReLU

        # ----------------------
        # Implicit normalization
        # ----------------------
        #self.normalize_flow = Normalize(mean=[1.4125, -0.9003], std=[41.2430, 41.1322])
        self.normalize_flow = Normalize(mean=[0.0, 0.0], std=[41.2430, 41.1322])

        # ----------------------------------------------------
        # Feature extractor encoder module for the LSTM module
        # ----------------------------------------------------

        activation = nn.Mish

        self.encoder_CNN = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, groups=in_channels),
            Conv(in_channels=self.in_channels, 
                 out_channels=16, 
                 kernel_size=7, 
                 stride=2, 
                 padding=3, 
                 activation=activation,
                 bias=False),
            ResidualConv(in_channels=16, out_channels=16, stride=2),
            ResidualConv(in_channels=16, out_channels=16, stride=2),
            ResidualConv(in_channels=16, out_channels=16, stride=2),
            ResidualConv(in_channels=16, out_channels=16, stride=2),
            ResidualConv(in_channels=16, out_channels=16, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1920, out_features=1024, bias=True),
            nn.Dropout(p=0.2),
            activation(inplace=True)
        )        

        # --------------------------------------------------
        # Blocks of the LSTM (Long Short Term Memory) module
        # --------------------------------------------------
        self.lstm_out_size = 512
        self.lstm1 = nn.LSTMCell(input_size=1024,
                                 hidden_size=self.lstm_out_size)

        self.lstm_states1 = (torch.zeros(self.batch_size, self.lstm_out_size).to(device), 
                             torch.zeros(self.batch_size, self.lstm_out_size).to(device))

        self.lstm2 = nn.LSTMCell(input_size=self.lstm_out_size,
                                 hidden_size=self.lstm_out_size)

        self.lstm_states2 = (torch.zeros(self.batch_size, self.lstm_out_size).to(device), 
                             torch.zeros(self.batch_size, self.lstm_out_size).to(device))

        # -------------------------
        # Odometry estimator module
        # -------------------------
        self.translation_regressor = Regressor_MLP(in_features=self.lstm_out_size, 
                                                   out_features=3, 
                                                   activation=activation, 
                                                   bias=False)

        self.rotation_regressor = Regressor_MLP(in_features=self.lstm_out_size, 
                                                out_features=3, 
                                                activation=activation, 
                                                bias=False)


    def forward(self, flows : torch.Tensor):
        """
        The "call" function of the odometry estimator.

        :param flows: Input optical flow values
        :type flows: torch.Tensor
        :return: The estimated euler angle rotation vector and estimated translation vector
        :rtype: torch.Tensor
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
        self.lstm_states1 = self.lstm1(features, self.lstm_states1)
        self.lstm_states2 = self.lstm2(self.lstm_states1[0], self.lstm_states2)
        lstm_out = self.lstm_states2[0]
        #log("LSTM out:", lstm_out.shape)

        # ----------------------------------
        # Odometry module translation branch
        # ----------------------------------
        translations = self.translation_regressor(lstm_out)
        rotations = self.rotation_regressor(lstm_out)

        return rotations, translations


    def reset_lstm(self):
        self.lstm_states1 = (torch.zeros(self.batch_size, self.lstm_out_size).to(self.device), 
                            torch.zeros(self.batch_size, self.lstm_out_size).to(self.device))
        self.lstm_states2 = (torch.zeros(self.batch_size, self.lstm_out_size).to(self.device), 
                            torch.zeros(self.batch_size, self.lstm_out_size).to(self.device))


    # TODO implement .to() for lstm_states


class Regressor_MLP(nn.Module):
    """
    Regressor Multilayer Perceptron head of the odometry module.
    
    :param in_features: Input feature size
    :param out_features: Output feature size
    :param activation: The type of activation function used in the regressor
    """
    def __init__(
        self, 
        in_features : int, 
        out_features : int, 
        activation = nn.PReLU, 
        bias : bool = True
    ) -> None:
        super(Regressor_MLP, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.Dropout(p=0.2),
            activation(inplace=True),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.Dropout(p=0.2),
            activation(inplace=True),
            nn.Linear(in_features=64, out_features=out_features, bias=bias)
        )


    def forward(self, x):
        """
        The "call" function of the regressor.

        :param x: The input from the LSTM modules
        :type x: torch.Tensor
        :return: The estimated rotation or translation
        :rtype: torch.Tensor
        """
        return self.regressor(x)
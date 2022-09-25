from GMA.core.network import RAFTGMA
import torch
from torch import nn
from general_layers.conv import Conv, ResidualConv
from general_layers.vit import ViT

from helpers import log


class CLVO(nn.Module):
    def __init__(self, batch_size=1, precomputed_flows=False, in_channels=5):
        super(CLVO, self).__init__()

        # ------------------------------
        # Storing init object parameters
        # ------------------------------
        
        self.batch_size = batch_size
        self.precomputed_flows = precomputed_flows
        self.in_channels = in_channels

        self.activation = nn.PReLU
        self.MLP_activation = nn.PReLU

        # ----------------------------------------------------
        # Feature extractor encoder module for the LSTM module
        # ----------------------------------------------------

        DownSample = Conv
        activation = nn.Mish

        #self.encoder_CNN = nn.Sequential(
        #    DownSample(in_channels=2, out_channels=16, kernel_size=[5, 5], stride=[1, 1], padding=2, activation=activation),
        #    DownSample(in_channels=16, out_channels=16, kernel_size=[5, 5], stride=[5, 5], padding=0, activation=activation),
        #    DownSample(in_channels=16, out_channels=16, kernel_size=[5, 5], stride=[5, 5], padding=0, activation=activation),
        #    DownSample(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[3, 3], padding=0, activation=activation),
        #    DownSample(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=0, activation=activation),
        #    nn.Flatten(),
        #    nn.Linear(in_features=1344, out_features=1024),
        #    nn.Dropout(0.2),
        #    activation(inplace=True)
        #)

        self.encoder_CNN = nn.Sequential(
            Conv(in_channels=2, out_channels=4, kernel_size=7, stride=1),
            ResidualConv(in_channels=4, out_channels=8),
            ResidualConv(in_channels=8, out_channels=16, stride=2),
            ResidualConv(in_channels=16, out_channels=32, stride=2),
            ResidualConv(in_channels=32, out_channels=32, stride=2),
            ResidualConv(in_channels=32, out_channels=32, stride=2)
        )

        self.vit = ViT(in_channels=32, device="cuda")

        # --------------------------------------------------
        # Blocks of the LSTM (Long Short Term Memory) module
        # --------------------------------------------------
        self.lstm_out_size = 256
        self.lstm_t = nn.LSTMCell(input_size=256,
                                  hidden_size=self.lstm_out_size)

        self.lstm_states_t = (torch.zeros(self.batch_size, self.lstm_out_size).to('cuda'), 
                              torch.zeros(self.batch_size, self.lstm_out_size).to('cuda'))
 
        self.lstm_r = nn.LSTMCell(input_size=256,
                                  hidden_size=self.lstm_out_size)

        self.lstm_states_r = (torch.zeros(self.batch_size, self.lstm_out_size).to('cuda'), 
                              torch.zeros(self.batch_size, self.lstm_out_size).to('cuda'))

        # -------------------------
        # Odometry estimator module
        # -------------------------
        self.translation_regressor = Regressor_MLP(in_features=self.lstm_out_size, out_features=3, activation=activation)
        self.rotation_regressor = Regressor_MLP(in_features=self.lstm_out_size, out_features=3, activation=activation)


    def forward(self, *args):
        flows = args[0]
        #log("Flows: ", flows.shape)

        # ------------------
        # Extracted features
        # ------------------
        features = self.encoder_CNN(flows)
        #log("Encoder out: ", features.shape)
        vit_features = self.vit(features)
        #log("VIT encoder out: ", vit_features.shape)

        # ----------------------
        # Long Short Term Memory
        # ----------------------
        self.lstm_states_t = self.lstm_t(vit_features[:, 0, :], self.lstm_states_t)
        lstm_out_t = self.lstm_states_t[0]
        
        self.lstm_states_r = self.lstm_r(vit_features[:, 1, :], self.lstm_states_r)
        lstm_out_r = self.lstm_states_r[0]
        #lstm_out = lstm_out.squeeze()
        #log("LSTM out:", lstm_out.shape)

        # ----------------------------------
        # Odometry module translation branch
        # ----------------------------------
        #log("MLP input shape", mlp_input.shape)
        translations = self.translation_regressor(lstm_out_t)
        rotations = self.rotation_regressor(lstm_out_r)

        return rotations, translations


    def reset_lstm(self):
        self.lstm_states_t = (torch.zeros(self.batch_size, self.lstm_out_size).to('cuda'), 
                            torch.zeros(self.batch_size, self.lstm_out_size).to('cuda'))
        self.lstm_states_r = (torch.zeros(self.batch_size, self.lstm_out_size).to('cuda'), 
                            torch.zeros(self.batch_size, self.lstm_out_size).to('cuda'))


class Regressor_MLP(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.PReLU) -> None:
        super(Regressor_MLP, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            activation(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),
            activation(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=out_features)
            
        )

        #self.regressor.apply(init_weights)

    def forward(self, x):
        return self.regressor(x)
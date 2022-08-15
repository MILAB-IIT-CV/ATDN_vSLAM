from GMA.core.network import RAFTGMA
import torch
from torch import nn
from general_layers.conv import Conv, ResidualConv

from helpers import log


class CLVO(nn.Module):
    def __init__(self, args, precomputed_flows=False, in_channels=5):
        super(CLVO, self).__init__()

        # ------------------------------
        # Storing init object parameters
        # ------------------------------

        self.precomputed_flows = precomputed_flows
        self.in_channels = in_channels

        self.activation = nn.PReLU
        self.MLP_activation = nn.PReLU

        # ---------------------------------------
        # Advanced optical flow extractor network
        # ---------------------------------------

        #if not self.precomputed_flows:
        #    self.raft_gma = RAFTGMA(args).to(args.device)
        #    weights = torch.load(args.model, map_location=args.device)
        #    self.raft_gma.load_state_dict(weights)
        #    for param in self.raft_gma.parameters():
        #        param.requires_grad = False

        # ----------------------------------------------------
        # Feature extractor encoder module for the LSTM module
        # ----------------------------------------------------

        DownSample = Conv
        activation = nn.Mish

        self.encoder_CNN = nn.Sequential(
            DownSample(in_channels=8, out_channels=16, kernel_size=[5, 5], stride=[1, 1], padding=2, activation=activation),
            DownSample(in_channels=16, out_channels=16, kernel_size=[5, 5], stride=[5, 5], padding=0, activation=activation),
            DownSample(in_channels=16, out_channels=16, kernel_size=[5, 5], stride=[5, 5], padding=0, activation=activation),
            DownSample(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[3, 3], padding=0, activation=activation),
            DownSample(in_channels=32, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=0, activation=activation),
            nn.Flatten(),
            nn.Linear(in_features=1344, out_features=1024),
            nn.Dropout(0.2),
            activation(inplace=True)
        )

        # --------------------------------------------------
        # Blocks of the LSTM (Long Short Term Memory) module
        # --------------------------------------------------

        self.lstm = nn.LSTM(input_size=1024,
                            hidden_size=512,
                            num_layers=4,
                            dropout=0.2,
                            batch_first=True)

        # -------------------------
        # Odometry estimator module
        # -------------------------

        # Translation branch
        self.translation_regressor = Regressor_MLP(in_features=512, out_features=3, activation=activation)

        # Rotation branch
        self.rotation_regressor = Regressor_MLP(in_features=512, out_features=3, activation=activation)


    def forward(self, *args):
        flows = args[0]
        #log("Flows: ", flows.shape)

        # ------------------
        # Extracted features
        # ------------------
        
        features = self.encoder_CNN(flows).unsqueeze(1)
        #log("Encoder out: ", features.shape)

        # ----------------------
        # Long Short Term Memory
        # ----------------------
        
        lstm_out, (hn, cn) = self.lstm(features)
        lstm_out = lstm_out.squeeze()
        #log("LSTM out:", lstm_out.shape)

        # ----------------------------------
        # Odometry module translation branch
        # ----------------------------------

        translations = self.translation_regressor(lstm_out)
        rotations = self.rotation_regressor(lstm_out)

        return rotations, translations



class Regressor_MLP(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.PReLU) -> None:
        super(Regressor_MLP, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            #nn.PReLU(),
            activation(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),
            #nn.PReLU(),
            activation(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=out_features)
            
        )

        #self.regressor.apply(init_weights)

    def forward(self, x):
        return self.regressor(x)
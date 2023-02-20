import torch
from torch import nn
from torchvision.transforms import Normalize

from ..layers.conv import Conv, ResidualConv
from ..layers.linear import Linear
from ..utils.helpers import log, ShapeLogLayer


class TVO(nn.Module):
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
        self.suffix = ''

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
        use_dropout = True
        
        self.embedding_dim = 128
        p = 16

        self.encoder_CNN = nn.Sequential(
            #ResidualConv(in_channels=in_channels, out_channels=16, stride=2, activation=activation),
            ResidualConv(in_channels=in_channels, out_channels=2, stride=2, activation=activation),
            nn.Conv2d(in_channels=in_channels, out_channels=self.embedding_dim, kernel_size=p, stride=p),
            nn.Flatten(start_dim=2)
        )        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4, dim_feedforward=1024, activation=nn.Mish())
        self.t_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
        self.positional_embedding = nn.Parameter(torch.rand((self.embedding_dim, 500), device="cuda"), requires_grad=True).unsqueeze(0)

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2)
        self.lstm_state = None
        # TODO make another transformer to encode dependencies between features
        # --------------------------------------
        # MLP heads for rotation and translation
        # --------------------------------------
        self.translation_regressor = nn.Sequential(
                Linear(in_features=self.embedding_dim, out_features=128, activation=activation, dropout=use_dropout, norm=True),
                Linear(in_features=128, out_features=64, activation=activation, dropout=use_dropout, norm=True),
                nn.Linear(in_features=64, out_features=3, bias=False)
        )

        self.rotation_regressor = nn.Sequential(
                Linear(in_features=self.embedding_dim, out_features=128, activation=activation, dropout=use_dropout, norm=True),
                Linear(in_features=128, out_features=64, activation=activation, dropout=use_dropout, norm=True),
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
        B, S, C, H, W = flows.size()
        flows = flows.view(B*S, C, H, W)
        normalized_flows = self.normalize_flow(flows)
        # ------------------
        # Extracted features
        # ------------------
        features = self.encoder_CNN(normalized_flows)
        #log("Encoder out: ", features.shape)

        # -------------------
        # Transformer Encoder
        # -------------------
        embeddings = features + self.positional_embedding[:, :, :features.shape[-1]]
        embeddings = embeddings.permute(2, 0, 1)
        tr_out = self.t_encoder(embeddings)[0] # TODO change later for second transformer
        #log("Transformer output shape: ", tr_out.shape)
        tr_out = tr_out.view(B, S, self.embedding_dim).permute(1, 0, 2)
        
        #lstm_out, self.lstm_state = self.lstm(tr_out, self.lstm_state)
        lstm_out, _ = self.lstm(tr_out)
        lstm_out = lstm_out.permute(1, 0, 2)

        # ----------------------------------
        # Odometry module translation branch
        # ----------------------------------
        rotations = self.rotation_regressor(lstm_out)
        translations = self.translation_regressor(lstm_out)

        return rotations, translations

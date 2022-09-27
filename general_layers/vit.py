import torch
from torch import nn
from helpers import log
from general_layers.conv import Conv


class ViT(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=16*16, device='cuda') -> None:
        super(ViT, self).__init__()
        embedding_num = 95
        self.patch_size = patch_size

        self.patch_conv = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=embedding_dim, 
                                    kernel_size=patch_size, 
                                    stride=patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, 
                                                   nhead=4, 
                                                   dropout=0.2,
                                                   activation="gelu",
                                                   dim_feedforward=1024)
        
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                                 num_layers=4)

        # TODO change for ability to process variable size input
        self.positional_embedding = nn.Parameter(torch.rand((embedding_dim, embedding_num)), requires_grad=True).unsqueeze(0).to(device)


    def forward(self, input):
        # Basic patch encoding
        #patches = input.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        #b, c, _, _, pw, ph = patches.shape 
        #embeddings = patches.reshape((b, -1, pw, ph)).flatten(2)

        # Input shape: Batch, Channel, H, W
        # Embedding shape: sequence_dim, Batch, embedding_dim (self.encoder_layer is in batch_first mode)
        #log("Input: ", input.shape)
        embeddings = self.patch_conv(input).flatten(2)
        embeddings = embeddings + self.positional_embedding
        #log("Embedding1: ", embeddings.shape)
        embeddings = embeddings.permute((2, 0, 1))
        #log("Embedding2: ", embeddings.shape)
        
        # Positional encoding

        # Result shape: Batch, embedding_dim, sequence_dim
        result = self.transformer(embeddings)
        #log("Result1: ", result.shape)
        result = result.permute((1, 0, 2))
        #log("Result2: ", result.shape)
        return result
import torch
from torch import nn
from helpers import log


class ViT(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=16*16, device='cuda') -> None:
        super(ViT, self).__init__()
        embedding_num = 95
        self.patch_size = patch_size

        self.patch_conv = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, 
                                                   nhead=4, 
                                                   dropout=0.2,
                                                   activation="gelu",
                                                   dim_feedforward=1024,
                                                   batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)

        self.positional_embedding = nn.Parameter(torch.rand(embedding_num, embedding_dim), requires_grad=True).unsqueeze(0).to(device)
        self.norm = nn.BatchNorm1d(num_features=embedding_num)


    def forward(self, input):
        #patches = input.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        #b, c, _, _, pw, ph = patches.shape 
        #embeddings = patches.reshape((b, -1, pw, ph)).flatten(2)
        embeddings = self.patch_conv(input).flatten(2).permute((0, 2, 1))
        #log("Embedding shape: ", embeddings.shape)
        embeddings = embeddings + self.positional_embedding 
        embeddings = self.norm(embeddings)

        result = self.transformer(embeddings)
        return result
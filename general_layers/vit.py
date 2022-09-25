import torch
from torch import nn
from helpers import log


class ViT(nn.Module):
    def __init__(self, in_channels, patch_size=16, embedding_dim=16*16, device='cuda') -> None:
        super(ViT, self).__init__()
        embedding_num = 128
        self.patch_size = patch_size

        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, 
                                                   nhead=4, 
                                                   dropout=0.2,
                                                   dim_feedforward=1024,
                                                   batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)

        self.positional_embedding = nn.Parameter(torch.rand(embedding_num, embedding_dim), requires_grad=True).unsqueeze(0).to(device)


    def forward(self, input):
        patches = input.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        b, c, _, _, pw, ph = patches.shape 
        embeddings = patches.reshape((b, -1, pw, ph)).flatten(2)

        embeddings = embeddings + self.positional_embedding 

        result = self.transformer(embeddings)
        return result
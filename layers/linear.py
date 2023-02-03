from torch import nn
import torch


class Linear(nn.Module):
    def __init__(
        self,
        in_features : int,
        out_features : int,
        bias : bool = True,
        activation = nn.PReLU,
        dropout : bool = True,
        norm : bool = False
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = nn.Identity()

        if norm is True:
            self.norm = nn.LayerNorm(normalized_shape=out_features)
        else:
            self.norm = nn.Identity()

        if dropout:
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = nn.Identity()


    def forward(self, input):
        x = self.linear(input)
        x = self.activation(x)
        x = self.norm(x)
        # TODO if confilct happens change order
        x = self.dropout(x)

        return x
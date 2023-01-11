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


class SimLin(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation=nn.PReLU,
        norm : bool = False,
        dropout : bool = True,
        bias=True
    ) -> None:
        super().__init__()
        
        assert ((out_features % 2) == 0), "Out features should be multiple of 2"
        self.out_features = out_features

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

        if norm:
            self.norm = nn.BatchNorm1d(out_features)
        else:
            self.norm = nn.Identity()

        if dropout:
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = nn.Identity()

        self.activation_pos = activation()
        self.activation_neg = activation()
        self.out_lin = nn.Linear(in_features=out_features*2, out_features=out_features)


    def forward(self, input):
        # Linear & split
        x = self.linear(input)
        
        # Symmetric activation
        x_pos = self.activation_pos(x)
        x_neg = (-1.0)*self.activation_neg((-1.0)*x)
        x = torch.cat([x_pos, x_neg], dim=1)
        x = self.out_lin(x)
        # Regularization
        x = self.norm(x)
        x = self.dropout(x)

        return x

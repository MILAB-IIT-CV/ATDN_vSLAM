import torch
import torch.nn.functional as F


def VAE_loss( input:torch.tensor, 
              predictions:torch.tensor,
              mu:torch.tensor,
              log_var:torch.tensor,
              beta=1) -> torch.float32:

    reconstruction_loss = F.mse_loss(predictions, input)
    KLD_loss = (-0.5*(1 + log_var - (mu**2) - log_var.exp()).sum(dim=-1)).mean(0)

    loss = reconstruction_loss + beta*KLD_loss

    return loss, KLD_loss, reconstruction_loss

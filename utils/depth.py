import numpy as np
import torch


def read_calib(path: str, include_rect : bool = False):
    """
    Read calibration matrix from file, optionally including rectification data.

    :param path: Path to calibration
    :param include_rect: Wheter to include rectification data in the matrix
    """
    calib = np.loadtxt(path, dtype=str)
    calib = calib[1][1:].astype(np.float32)
    calib = torch.from_numpy(calib).view((3, 4))
    if include_rect:
        calib = torch.cat([calib, torch.tensor([[0, 0, 0, 1]])])
    else:
        calib = calib[:, :-1]

    return calib


def project_depth(depth : torch.Tensor, calib : torch.Tensor, device : str = "cuda:0"):
    """
    Project depth to 3D space with calibration data.
    
    :param depth: Depth image to project into 3D
    :param calib: Calibration matrix for the image
    :param device: Device to map depth to
    """
    height, width = depth.shape[-2], depth.shape[-1]
    
    tensor_u = torch.arange(0, width, 1).repeat((height, 1)).to(device)
    tensor_v = torch.arange(0, height, 1).repeat((width, 1)).permute(1, 0).to(device)

    h_u = tensor_u*depth
    h_v = tensor_v*depth

    homogenous = torch.stack([h_u, h_v, depth], dim=-1)
    
    flattened_h = homogenous.view((height*width, 3)).permute(1, 0)
    calibi = torch.inverse(calib)
    projected = torch.matmul(calibi, flattened_h).permute(1, 0)
    projected = projected.view((height, width, 3)).permute(2, 0, 1)
    
    return projected
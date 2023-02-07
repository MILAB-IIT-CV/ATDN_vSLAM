from os import path
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision.io import read_image
import cv2
import glob


from utils.helpers import log
from utils.transforms import matrix2euler


class ColorDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        sequence="00",
        hls = False,
        division = 1
    ) -> None:
        super().__init__()

        self.division = division
        self.hls = hls

        self.data_path = path.join(data_path, 'dataset', 'sequences', sequence, 'image_2')
        self.len = len(glob.glob(path.join(self.data_path, "*.png")))
        if division > 1:
            temp_len = self.len//division
            if (temp_len-1)*division > self.len-1:
                temp_len = temp_len-1
            self.len = temp_len


    def __len__(self):
        return self.len


    def __getitem__(self, index):
        index = index*self.division
        str_index = str(index)
        zero_len = (6-len(str_index))
        filename = zero_len*'0' + str_index
        file_path = path.join(self.data_path, filename + ".png")
        
        image = cv2.imread(file_path)
        if self.hls:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        return image


class DoubleColorDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        sequence,
        division = 1
    ) -> None:
        super().__init__()

        self.division = division

        self.data_path = path.join(data_path, 'dataset', 'sequences', sequence, 'image_2')
        self.len = len(glob.glob(path.join(self.data_path, "*.png")))
        if division > 1:
            temp_len = self.len//division
            if (temp_len-1)*division > self.len-1:
                temp_len = temp_len-1
            self.len = temp_len


    def __len__(self):
        return self.len


    def __getitem__(self, index):
        index = index*self.division
        str_index = str(index)
        zero_len = (6-len(str_index))
        filename = zero_len*'0' + str_index
        file_path = path.join(self.data_path, filename + ".png")

        image = cv2.imread(file_path)

        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hls = torch.from_numpy(hls).float().permute(2, 0, 1)
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)

        return hls, rgb



class DepthDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        sequence,
        division = 1
    ) -> None:
        super().__init__()

        self.division = division

        self.data_path = path.join(data_path, 'dataset', 'depths2', sequence)
        self.len = len(glob.glob(path.join(self.data_path, "*.pt")))
        if division > 1:
            temp_len = self.len//division
            if (temp_len-1)*division > self.len-1:
                temp_len = temp_len-1
            self.len = temp_len


    def __len__(self):
        return self.len


    def __getitem__(self, index):
        index = index*self.division
        str_index = str(index)
        zero_len = (6-len(str_index))
        filename = zero_len*'0' + str_index
        file_path = path.join(self.data_path, filename + ".pt")

        depth = torch.load(file_path).float().unsqueeze(0)

        return depth


class ColorDepthDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        sequence,
        hls = False,
        division = 1
    ) -> None:
        super().__init__()

        self.division = division
        self.hls = hls

        self.data_path_c = path.join(data_path, 'dataset', 'sequences', sequence, 'image_2')
        len_c = len(glob.glob(path.join(self.data_path_c, "*.png")))
        self.data_path_d = path.join(data_path, 'dataset', 'depths2', sequence)
        len_d = len(glob.glob(path.join(self.data_path_d, "*.pt")))

        assert len_c == len_d, "Different number of color and depth files!"
        self.len = len_c

        if division > 1:
            temp_len = self.len//division
            if (temp_len-1)*division > self.len-1:
                temp_len = temp_len-1
            self.len = temp_len


    def __len__(self):
        return self.len


    def __getitem__(self, index):
        index = index*self.division
        str_index = str(index)
        zero_len = (6-len(str_index))
        filename = zero_len*'0' + str_index

        depth_path = path.join(self.data_path_d, filename + ".pt")

        depth = torch.load(depth_path).float().unsqueeze(0)

        color_path = path.join(self.data_path_c, filename + ".png")
        image = cv2.imread(color_path)
        if self.hls:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        return depth, image
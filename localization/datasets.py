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

class KittiLocalizationDataset(data.Dataset):
    def __init__(self, data_path, sequence, simplify=False, simplification_rate=10):
        super().__init__()

        self.data_path = path.join(data_path, 'dataset')
        self.sequence = sequence
        self.simplify = simplify
        self.simplification_rate = simplification_rate

        self.poses = np.loadtxt(path.join(self.data_path, 'poses', self.sequence+'.txt'))

        self.im_path = path.join(self.data_path, "sequences", sequence, "image_2")

    def __len__(self):
        if self.simplify:
            return int(self.poses.shape[0]/self.simplification_rate)
        else:
            return self.poses.shape[0]

    def __getitem__(self, index):
        
        if self.simplify:
            index = index*self.simplification_rate

        # Getting pose difference as rotation and translation vectors
        pose1 = self.poses[index, :]
        
        pose1 = torch.from_numpy(np.array([pose1[0:4], pose1[4:8], pose1[8:12]]))
        
        abs_rotation = pose1[:, :3]
        abs_translation = pose1[:, -1]
        
        orientation = matrix2euler(abs_rotation)
        position = abs_translation        
        
        im_file1 = '0'*(6-len(str(index))) + str(index) + ".png"
        im1 = self.load_image(path.join(self.im_path, im_file1))

        return im1, orientation, position


    def load_image(self, imfile):
        img = np.expand_dims(np.array(Image.open(imfile)).astype(np.uint8), -1)
        img = torch.from_numpy(img).squeeze().permute(2, 0, 1).float()
        return img[None].squeeze()


class MappingDataset(data.Dataset):
    def __init__(self, data_path, slam=False):
        super().__init__()

        self.slam = slam
        if slam:
            self.data_path = data_path
            self.im_path = path.join(self.data_path, "rgb")
            self.length = len(glob.glob(self.im_path + "/*.pth"))
        else:
            self.data_path = path.join(data_path, 'dataset')
            self.im_path = path.join(self.data_path, "rgb")
            self.length = len(glob.glob(self.im_path))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
                
        extension = '.pth' if self.slam else '.png'
        img_file = '0'*(6-len(str(index))) + str(index) + extension
        if self.slam:
            img = torch.load(path.join(self.im_path, img_file)).squeeze()
        else:
            img = self.load_image(path.join(self.im_path, img_file))

        return img


    def load_image(self, imfile):
        img = np.expand_dims(np.array(Image.open(imfile)).astype(np.uint8), -1)
        img = torch.from_numpy(img).squeeze().permute(2, 0, 1).float()
        return img[None].squeeze()


class ColorDataset(data.Dataset):
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
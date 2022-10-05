from os import path
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from scipy.spatial.transform import Rotation as R
import glob

from utils.helpers import log, matrix2euler

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


#    def load_image(self, imfile):
#        img = np.expand_dims(np.array(Image.open(imfile)).astype(np.uint8), -1)
#        img = torch.from_numpy(img).expand(-1, -1, 3).permute(2, 0, 1).float()
#        return img[None].to(DEVICE)

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



class KittiUnetDataset(data.Dataset):
    def __init__(self, data_path, sequence, offset=5):
        super().__init__()

        self.data_path = path.join(data_path, 'dataset')
        self.sequence = sequence
        self.offset = offset
        self.sigma = torch.ceil(torch.tensor(offset/2))

        self.poses = np.loadtxt(path.join(self.data_path, 'poses', self.sequence+'.txt'))

        self.im_path = path.join(self.data_path, "sequences", sequence, "image_2")

    def __len__(self):
        return int(self.poses.shape[0]/self.offset)-2*self.offset

    def __getitem__(self, index):
        
        # Getting pose difference as rotation and translation vectors
        index = self.offset+index*self.offset
        pose1 = self.poses[index, :]
        pose1 = torch.from_numpy(np.array([pose1[0:4], pose1[4:8], pose1[8:12]]))
        
        abs_rotation = pose1[:, :3]
        abs_translation = pose1[:, -1]
        
        orientation = matrix2euler(abs_rotation)
        position = abs_translation        

        im_file_true = '0'*(6-len(str(index))) + str(index) + ".png"
        im_true = self.load_image(path.join(self.im_path, im_file_true))

        #log("Center index: ", index)
        index = torch.normal(mean=index, std=self.sigma)
        index = torch.maximum(index, torch.tensor(0))
        index = int(torch.minimum(index, torch.tensor(self.poses.shape[0]-1)))
        #log("Random index: ", index)
        
        im_file_rand = '0'*(6-len(str(index))) + str(index) + ".png"
        im_rand = self.load_image(path.join(self.im_path, im_file_rand))

        return im_true, im_rand, orientation, position


    def load_image(self, imfile):
        img = np.expand_dims(np.array(Image.open(imfile)).astype(np.uint8), -1)
        img = torch.from_numpy(img).squeeze().permute(2, 0, 1).float()
        return img[None].squeeze()

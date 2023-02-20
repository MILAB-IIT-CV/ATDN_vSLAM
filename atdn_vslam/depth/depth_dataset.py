from os import path
import glob

import numpy as np
import torch
from torchvision.transforms import Resize
from torchvision.transforms.transforms import InterpolationMode
from torchvision.io import read_image

from GMA.core.utils.utils import InputPadder
from ..utils.transforms import line2matrix
from ..utils.depth import read_calib
from ..odometry.datasets import OdometryDataset


class OdometryDepthDataset(OdometryDataset):
    def __init__(
        self, 
        data_path : str, 
        sequences=['00'], 
        augment : bool = False, 
        sequence_length : int = 4, 
        device : str = 'cuda',
        pad : bool = False
    ) -> None:
        super(OdometryDepthDataset, self).__init__()

        self.sequences = sequences
        self.augment = 0.0 if augment else 1.0
        self.N = sequence_length
        self.device = device
        self.pad = pad

        self.data_path = path.join(data_path, 'dataset')
        self.sequence_lengths = [0]
        self.sequence_poses = []
        self.calibs = []

        # Checking type of sequences
        assert type(sequences) is list, "Sequences should be given as a list of strings!"
        for sequence in sequences:
            # Checking type and existence of each sequence
            assert type(sequence) is str, "Sequence should be given as a string!"
            if not path.isfile(path.join(self.data_path, "poses", sequence+".txt")):
                raise Exception("Sequence " + sequence + " doesn't exist!")
            
            im_path = path.join(self.data_path, "sequences", sequence, "image_2")
            sequence_length = len(glob.glob(im_path+"/*.png"))-self.N
            self.sequence_lengths.append(self.sequence_lengths[-1]+sequence_length) 
            
            pose = np.loadtxt(path.join(self.data_path, 'poses', sequence+'.txt'), dtype=np.double)
            self.sequence_poses.append(torch.from_numpy(pose))

            self.calibs.append(read_calib(path.join(self.data_path, "sequences",  sequence, "calib.txt")))

        self.sequence_lengths = self.sequence_lengths[1:]
        assert len(sequences) == len(self.sequence_lengths), "Sequence lengths are not as many as sequences"

        img = read_image(path.join(self.data_path, "sequences", '00', "image_2", "000000.png")).float()
        self.resize = Resize((img.shape[-2], img.shape[-1]))
        self.resize2 = Resize((img.shape[-2], img.shape[-1]), interpolation=InterpolationMode.NEAREST)
        self.padder = InputPadder(img.shape)


    def __len__(self):
        return self.sequence_lengths[-1]


    def __getitem__(self, index):
            reverse = (torch.rand(1) + self.augment) < 0.5
            
            sequence_index = 0
            index_offset = 0
            # Finding the sequence
            for i in range(len(self.sequence_lengths)):
                if index >= self.sequence_lengths[i]:
                    sequence_index = sequence_index+1
                    index_offset = self.sequence_lengths[i]

            index = index-index_offset

            # Getting pose difference as rotation and translation vectors
            poses_n = [self.sequence_poses[sequence_index][index+i, :] for i in range(0, self.N+1)]
            delta_transforms = [self.to_relative_matrix(poses_n[i], poses_n[i+1]) for i in range(0, (self.N))]
            delta_transforms = torch.stack(delta_transforms, dim=0).squeeze()

            # Generating image file names from index
            im_path = path.join(self.data_path, "sequences", self.sequences[sequence_index], "image_2")
            img_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".png" for i in range(0, self.N+1)]
            imgs = [self.resize(read_image(path.join(im_path, im_file)).float()) for im_file in img_files]
            if self.pad:
                imgs = [self.padder.pad(imgs[i])[0] for i in range(0, self.N+1)]
            imgs = torch.stack([imgs[i] for i in range(0, self.N+1)], dim=0)

            # Generating image file names from index
            mask_path = path.join(self.data_path, "skymasks", self.sequences[sequence_index])
            mask_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".pth" for i in range(0, self.N+1)]
            masks = [self.resize(torch.load(path.join(mask_path, mask_file)).unsqueeze(0).float()) for mask_file in mask_files]
            #masks = [torch.load(path.join(mask_path, mask_file)).unsqueeze(0) for mask_file in mask_files]
            masks = torch.stack([masks[i] for i in range(0, self.N+1)], dim=0).bool().squeeze()

            flow_path = path.join(self.data_path, "flows", self.sequences[sequence_index])
            # Generating flow file names from index
            flow_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".pt" for i in range(0, self.N)]
            flows = [torch.load(path.join(flow_path, flow_file)) for flow_file in flow_files]
            flows = torch.stack(flows, dim=0).squeeze()
            flows = self.resize(flows)
            
            return imgs, flows, delta_transforms, self.calibs[sequence_index], masks


    def to_relative_matrix(self, pose1, pose2):
        pose1 = line2matrix(pose1)
        inverted1 = torch.inverse(pose1)

        pose2 = line2matrix(pose2)
        
        delta_pose = torch.matmul(inverted1, pose2)

        return delta_pose
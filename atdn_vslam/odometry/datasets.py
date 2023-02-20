from os import path
import glob
import numpy as np
import torch
from torch.utils import data
import torchvision.io as io

from ..utils.transforms import matrix2euler, line2matrix, abs2rel
from tqdm import trange


class KittiOdometryDataset(data.Dataset):
    """
    Dataset for RGB image and absolute pose
    """
    def __init__(
        self,
        data_path : str,
        sequence : str
    ):
        super().__init__()
        
        self.poses = torch.from_numpy(np.loadtxt(path.join(data_path, "dataset", "poses", sequence+".txt")))
        self.rgb_files = sorted(glob.glob(path.join(data_path, "dataset", "sequences", sequence, "image_2/*.png")))

        assert len(self.poses) == len(self.rgb_files), "Number of poses and images are not the same"

    def __len__(self):
        return len(self.poses)


    def __getitem__(self, index):
        image = io.read_image(self.rgb_files[index])
        
        pose = line2matrix(self.poses[index])
        rot = matrix2euler(pose[:3, :3])
        tr = pose[:3, -1]

        return image, rot, tr


class FlowKittiDataset2(data.Dataset):
    """
    Dataset for 16 bit half precision optical flow and delta transforms
    """
    def __init__(
        self, 
        data_path : str, 
        sequences : list = ['00'], 
        augment = False, 
        sequence_length : int = 4
    ) -> None:
        super().__init__()

        self.sequences = sequences
        if type(augment) is bool:
            self.augment = 0 if augment else 1
        else:
            self.augment = augment
        self.N = sequence_length

        self.data_path = path.join(data_path, 'dataset')
        self.sequence_lengths = [0]
        self.sequence_poses = []

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

        self.sequence_lengths = self.sequence_lengths[1:]
        assert len(sequences) == len(self.sequence_lengths), "Sequence lengths are not as many as sequences"


    def __len__(self):
        return self.sequence_lengths[-1]


    def __getitem__(self, index):
        reverse = (self.augment + torch.rand(1)) < 0.5
        #log("Reverse flow augmentation: ", augment)

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

        if reverse:
            poses_n.reverse()
        delta_transforms = [abs2rel(poses_n[i], poses_n[i+1]) for i in range(0, (self.N))]

        delta_rotations = torch.stack([delta_transforms[i][0] for i in range(len(delta_transforms))])
        delta_translations = torch.stack([delta_transforms[i][1] for i in range(len(delta_transforms))])

        flow_path = path.join(self.data_path, "flows2", self.sequences[sequence_index])

        # Generating flow file names from index
        flow_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".pt" for i in range(0, self.N)]
        flows = [torch.load(path.join(flow_path, flow_file)) for flow_file in flow_files]
        flows = torch.cat(flows, dim=0)
        
        if flows.size()[-1] > 1232:
            diff = flows.size()[-1]-1232
            flows = flows[:, :, :, diff//2:-diff//2]
        flows = flows.squeeze()

        if reverse:
            flows = -1.0*flows
            if self.N > 1:
                flows = torch.flip(flows, dims=[0])

        return flows, delta_rotations, delta_translations


class FlowKittiDataset3(data.Dataset):
    """
    Dataset for 16 bit half precision optical flow loaded into CPU memory and delta transforms
    """
    def __init__(
        self, 
        data_path : str, 
        sequences : list = ['00'], 
        augment = False, 
        sequence_length : int = 4
    ) -> None:
        super().__init__()

        self.sequences = sequences
        if type(augment) is bool:
            self.augment = 0 if augment else 1
        else:
            self.augment = augment
        self.N = sequence_length

        self.data_path = path.join(data_path, 'dataset')
        self.sequence_lengths = [0]
        self.sequence_poses = []

        # Checking type of sequences
        assert type(sequences) is list, "Sequences should be given as a list of strings!"
        for sequence in sequences:
            # Checking type and existence of each sequence
            assert type(sequence) is str, "Sequence should be given as a string!"
            if not path.isfile(path.join(self.data_path, "poses", sequence+".txt")):
                raise Exception("Sequence " + sequence + " doesn't exist!")
            
            im_path = path.join(self.data_path, "sequences", sequence, "image_2")
            sequence_length = len(glob.glob(im_path+"/*.png")) - self.N
            self.sequence_lengths.append(self.sequence_lengths[-1] + sequence_length) 
            
            pose = np.loadtxt(path.join(self.data_path, 'poses', sequence+'.txt'), dtype=np.double)
            self.sequence_poses.append(torch.from_numpy(pose))

        self.sequence_lengths = self.sequence_lengths[1:]
        assert len(sequences) == len(self.sequence_lengths), "Sequence lengths are not as many as sequences"

        self.flows_all = []
        for sequence in sequences:
            seq_path = path.join(self.data_path, "flows2", sequence, "*.pt")
            files = sorted(glob.glob(seq_path))
            self.flows_all.append([])

            for i in trange(len(files)):
                A = torch.load(files[i], map_location='cpu')

                if A.size()[-1] > 1232:
                    diff = A.size()[-1]-1232
                    A = A[:, :, :, diff//2:-diff//2]

                self.flows_all[-1].append(A)
            self.flows_all[-1] = torch.cat(self.flows_all[-1], dim=0)


    def __len__(self):
        return self.sequence_lengths[-1]


    def __getitem__(self, index):
            reverse = (self.augment + torch.rand(1)) < 0.5
            #log("Reverse flow augmentation: ", augment)

            sequence_index = 0
            index_offset = 0
            # Finding the sequence
            for i in range(len(self.sequence_lengths)):
                if index >= self.sequence_lengths[i]:
                    sequence_index = sequence_index + 1
                    index_offset = self.sequence_lengths[i]

            index = index - index_offset

            # Getting pose difference as rotation and translation vectors
            poses_n = [self.sequence_poses[sequence_index][index+i, :] for i in range(0, self.N+1)]

            if reverse:
                poses_n.reverse()
            
            delta_transforms = [abs2rel(poses_n[i], poses_n[i+1]) for i in range(0, (self.N))]
            delta_rotations = torch.stack([delta_transforms[i][0] for i in range(len(delta_transforms))])
            delta_translations = torch.stack([delta_transforms[i][1] for i in range(len(delta_transforms))])

            flows = self.flows_all[sequence_index][index:index+self.N]
            if reverse:
                flows = (-1)*flows
                if self.N > 1:
                    flows = torch.flip(flows, dims=[0])
            
            return flows, delta_rotations, delta_translations

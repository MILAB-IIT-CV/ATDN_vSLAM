from os import path
import glob
import numpy as np
import torch
from torch.utils import data
from GMA.core.utils.utils import InputPadder
from torchvision.transforms import Resize
import torchvision.io as io

from helpers import matrix2euler, line2matrix, log


class OdometryDataset(data.Dataset):
    def __init__(self):
            super(OdometryDataset, self).__init__()
    

    def preprocess_poses_euler(self,  pose1, pose2):
        # Stacking the matrix rows stored in the lines of the array
        pose1 = line2matrix(pose1)
        inverted1 = torch.inverse(pose1)

        pose2 = line2matrix(pose2)
        
        delta_pose = torch.matmul(inverted1, pose2)
        delta_rot = delta_pose[:3, :3]
        delta_translation = delta_pose[:3, -1]

        delta_rotation = matrix2euler(delta_rot)

        return [delta_rotation, delta_translation]
        

class KittiOdometryDataset(OdometryDataset):
    def __init__(self, data_path, sequence, precomputed_flow=False, sequence_length=4, device='cuda'):
        super(KittiOdometryDataset, self).__init__()
        
        self.sequence = sequence
        self.precomputed_flow = precomputed_flow
        self.N = sequence_length
        self.device = device
        self.resize = Resize((376, 1248))

        self.data_path = path.join(data_path, 'dataset')

        self.im_path = path.join(self.data_path, "sequences", sequence, "image_2")
        self.flow_path = path.join(self.data_path, "flows", sequence)
        self.poses = np.loadtxt(path.join(self.data_path, 'poses', self.sequence+'.txt'), dtype=np.double)
        self.poses = torch.from_numpy(self.poses)

        im_file = "000000.png"
        img = io.read_image(path.join(self.im_path, im_file)).float()
        self.padder = InputPadder(img.shape)
        self.len = len(glob.glob(self.im_path+"/*.png"))-self.N

    def __len__(self):
        return self.len

    def __getitem__(self, index):
            
            # Getting pose difference as rotation and translation vectors
            poses_n = [self.poses[index+i, :] for i in range(0, self.N+1)]
            delta_transforms = [super(KittiOdometryDataset, self).preprocess_poses_euler(poses_n[i], poses_n[i+1]) for i in range(0, (self.N))]

            # Getting the image file names
            img_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".png" for i in range(0, self.N+1)]

            imgs = torch.stack([self.padder.pad(io.read_image(path.join(self.im_path, im_file)).float())[0] for im_file in img_files], dim=0)
            #imgs1 = torch.stack([imgs[i] for i in range(0, (self.N-1))], dim=0)
            #imgs2 = torch.stack([imgs[i] for i in range(1, self.N)], dim=0)
            imgs = torch.stack([imgs[i] for i in range(0, self.N+1)], dim=0)

            delta_rotations = [delta_transforms[i][0] for i in range(len(delta_transforms))]
            delta_translations = [delta_transforms[i][1] for i in range(len(delta_transforms))]
            delta_rotations = torch.stack(delta_rotations)
            delta_translations = torch.stack(delta_translations)


            if not self.precomputed_flow:
                #return imgs1, imgs2, delta_rotations, delta_translations
                return imgs, delta_rotations, delta_translations
            else:
                flow_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".pt" for i in range(0, self.N)]
                flows = [torch.load(path.join(self.flow_path, flow_file)) for flow_file in flow_files]
                flows = torch.stack(flows, dim=0)
                
                return imgs, flows, delta_rotations, delta_translations


class CustomKittiOdometryDataset(OdometryDataset):
    def __init__(self, data_path, sequences=['00'], precomputed_flow=False, sequence_length=4, device='cuda'):
        super(CustomKittiOdometryDataset, self).__init__()

        self.sequences = sequences
        self.precomputed_flow = precomputed_flow
        self.N = sequence_length
        self.device = device

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

        img = io.read_image(path.join(self.data_path, "sequences", '00', "image_2", "000000.png")).float()
        self.resize = Resize((img.shape[-2], img.shape[-1]))
        self.padder = InputPadder(img.shape)


    def __len__(self):
        return self.sequence_lengths[-1]


    def __getitem__(self, index):
            #actual_time = time.time()

            sequence_index = 0
            index_offset = 0
            # Finding the sequence
            for i in range(len(self.sequence_lengths)):
                if index >= self.sequence_lengths[i]:
                    sequence_index = sequence_index+1
                    index_offset = self.sequence_lengths[i]

            index = index-index_offset

            #log("Sequence_search: ", time.time()-actual_time)
            #actual_time = time.time()

            # Getting pose difference as rotation and translation vectors
            poses_n = [self.sequence_poses[sequence_index][index+i, :] for i in range(0, self.N+1)]
            delta_transforms = [super(CustomKittiOdometryDataset, self).preprocess_poses_euler(poses_n[i], poses_n[i+1]) for i in range(0, (self.N))]
            delta_rotations = [delta_transforms[i][0] for i in range(len(delta_transforms))]
            delta_translations = [delta_transforms[i][1] for i in range(len(delta_transforms))]
            delta_rotations = torch.stack(delta_rotations)
            delta_translations = torch.stack(delta_translations)
            
            #log("Transforms: ", time.time()-actual_time)
            #actual_time = time.time()

            # Generating image file names from index
            im_path = path.join(self.data_path, "sequences", self.sequences[sequence_index], "image_2")
            img_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".png" for i in range(0, self.N+1)]
            imgs = torch.stack([self.padder.pad(io.read_image(path.join(im_path, im_file)).float())[0] for im_file in img_files], dim=0)
            imgs = torch.stack([imgs[i] for i in range(0, self.N+1)], dim=0)
            imgs = self.resize(imgs)

            #log("Image load: ", time.time()-actual_time)
            #actual_time = time.time()

            if not self.precomputed_flow:
                return imgs, delta_rotations, delta_translations
            else:
                flow_path = path.join(self.data_path, "flows", self.sequences[sequence_index])
                # Generating flow file names from index
                flow_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".pt" for i in range(0, self.N)]
                flows = [torch.load(path.join(flow_path, flow_file)) for flow_file in flow_files]
                flows = torch.stack(flows, dim=0).squeeze()
                flows = self.resize(flows)

                #log("Flow fload: ", time.time()-actual_time)
                #actual_time = time.time()
                
                return imgs, flows, delta_rotations, delta_translations


class FlowKittiDataset(OdometryDataset):
    def __init__(self, data_path, sequences=['00'], precomputed_flow=False, sequence_length=4, device='cuda'):
        super(FlowKittiDataset, self).__init__()

        self.sequences = sequences
        self.precomputed_flow = precomputed_flow
        self.N = sequence_length
        self.device = device

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

        img = io.read_image(path.join(self.data_path, "sequences", '00', "image_2", "000000.png")).float()
        self.resize = Resize((img.shape[-2], img.shape[-1]))
        self.padder = InputPadder(img.shape)


    def __len__(self):
        return self.sequence_lengths[-1]


    def __getitem__(self, index):
            #actual_time = time.time()

            sequence_index = 0
            index_offset = 0
            # Finding the sequence
            for i in range(len(self.sequence_lengths)):
                if index >= self.sequence_lengths[i]:
                    sequence_index = sequence_index+1
                    index_offset = self.sequence_lengths[i]

            index = index-index_offset

            #log("Sequence_search: ", time.time()-actual_time)
            #actual_time = time.time()

            # Getting pose difference as rotation and translation vectors
            poses_n = [self.sequence_poses[sequence_index][index+i, :] for i in range(0, self.N+1)]
            delta_transforms = [super(FlowKittiDataset, self).preprocess_poses_euler(poses_n[i], poses_n[i+1]) for i in range(0, (self.N))]
            delta_rotations = [delta_transforms[i][0] for i in range(len(delta_transforms))]
            delta_translations = [delta_transforms[i][1] for i in range(len(delta_transforms))]
            delta_rotations = torch.stack(delta_rotations)
            delta_translations = torch.stack(delta_translations)

            #log("Image load: ", time.time()-actual_time)
            #actual_time = time.time()

            flow_path = path.join(self.data_path, "flows", self.sequences[sequence_index])
            # Generating flow file names from index
            flow_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".pt" for i in range(0, self.N)]
            flows = [torch.load(path.join(flow_path, flow_file)) for flow_file in flow_files]
            flows = torch.stack(flows, dim=0).squeeze()
            flows = self.resize(flows)

            #log("Flow fload: ", time.time()-actual_time)
            #actual_time = time.time()
                
            return flows, delta_rotations, delta_translations

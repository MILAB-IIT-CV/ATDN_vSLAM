from os import path
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from scipy.spatial.transform import Rotation as R
from GMA.core.utils.utils import InputPadder
from torchvision.transforms import Resize

from helpers import log, matrix2euler, euler2matrix


class KittiOdometryDataset(data.Dataset):
    def __init__(self, data_path, sequence, precomputed_flow=False, sequence_length=4, device='cuda'):
        super(KittiOdometryDataset, self).__init__()
        
        self.sequence = sequence
        self.precomputed_flow = precomputed_flow
        self.N = sequence_length+1
        self.device = device
        self.resize = Resize((376, 1248))

        self.data_path = path.join(data_path, 'dataset')

        self.im_path = path.join(self.data_path, "sequences", sequence, "image_2")
        self.flow_path = path.join(self.data_path, "flows", sequence)
        self.poses = np.loadtxt(path.join(self.data_path, 'poses', self.sequence+'.txt'), dtype=np.double)

        im_file = "000000.png"
        img = self.load_image(path.join(self.im_path, im_file))
        self.padder = InputPadder(img.shape)
        self.len = len(glob.glob(self.flow_path+"/*.pt"))

    def __len__(self):
        #return self.poses.shape[0]-self.N
        len = self.len-self.N
        return len

    def __getitem__(self, index):
            
            # Getting pose difference as rotation and translation vectors
            poses_n = [self.poses[index+i, :] for i in range(0, self.N)]
            delta_transforms = [self.preprocess_poses_euler(poses_n[i], poses_n[i+1]) for i in range(0, (self.N-1))]

            # Getting the image file names
            img_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".png" for i in range(0, self.N)]

            imgs = torch.stack([self.padder.pad(self.load_image(path.join(self.im_path, im_file)))[0] for im_file in img_files], dim=0)
            imgs1 = torch.stack([imgs[i] for i in range(0, (self.N-1))], dim=0)
            imgs2 = torch.stack([imgs[i] for i in range(1, self.N)], dim=0)

            delta_rotations = [delta_transforms[i][0] for i in range(len(delta_transforms))]
            delta_translations = [delta_transforms[i][1] for i in range(len(delta_transforms))]
            delta_rotations = torch.stack(delta_rotations)
            delta_translations = torch.stack(delta_translations)


            if not self.precomputed_flow:
                return imgs1, imgs2, delta_rotations, delta_translations
            else:
                flow_files = ['0'*(6-len(str(index+i))) + str(index+i) + ".pt" for i in range(0, self.N-1)]
                flows = [torch.load(path.join(self.flow_path, flow_file)) for flow_file in flow_files]
                flows = torch.stack(flows, dim=0)
                
                return imgs1, imgs2, flows, delta_rotations, delta_translations



    def load_image(self, imfile):
        img = np.expand_dims(np.array(Image.open(imfile)).astype(np.uint8), -1)
        img = torch.from_numpy(img).squeeze().permute(2, 0, 1).float()
        return img[None]


    def preprocess_poses_euler(self,  pose1, pose2):
        # Stacking the matrix rows stored in the lines of the array
        pose1 = torch.from_numpy(np.array([pose1[0:4], pose1[4:8], pose1[8:12]]))
        pose1 = torch.cat([pose1, torch.tensor([[0, 0, 0, 1]])], dim=0)
        inverted1 = torch.inverse(pose1)

        pose2 = torch.from_numpy(np.array([pose2[0:4], pose2[4:8], pose2[8:12]]))
        pose2 = torch.cat([pose2, torch.tensor([[0, 0, 0, 1]])], dim=0)
        
        delta_pose = torch.matmul(inverted1, pose2)

        delta_rot = delta_pose[:3, :3]
        delta_translation = delta_pose[:3, -1]

        delta_rotation = matrix2euler(delta_rot)


        return [delta_rotation, delta_translation]
        


    def prepocess_poses_euler2(self,  pose1, pose2):
        # Stacking the matrix rows stored in the lines of the array
        pose1 = np.array([pose1[0:4], pose1[4:8], pose1[8:12]])
        pose2 = np.array([pose2[0:4], pose2[4:8], pose2[8:12]])
        
        # Getting rotation matrix
        abs_rotation1 = pose1[:, :3]
        abs_rotation2 = pose2[:, :3]
        
        # Getting position vectors
        translation1 = pose1[:, -1]
        translation2 = pose2[:, -1]
        # Converting rotation matrix to euler angle vector
        abs_rotation1 = torch.from_numpy(abs_rotation1)
        abs_rotation2 = torch.from_numpy(abs_rotation2)
        # Calculating differences         
        #delta_rotation = matrix2euler(torch.matmul(abs_rotation1, abs_rotation2))
        delta_rotation = matrix2euler(torch.matmul(abs_rotation2, torch.transpose(abs_rotation1, 0, 1)))
        
        delta_translation = torch.from_numpy(np.array(translation2 - translation1))
        #delta_translation = torch.from_numpy(abs_rotation1.inv().apply(delta_translation))
        delta_translation = torch.matmul(torch.transpose(abs_rotation1, 0, 1), delta_translation)
        return [delta_rotation, delta_translation]

    
    
import os
import glob
import copy
from tqdm import trange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
import torchvision.transforms.functional as TF

from GMA.core.network import RAFTGMA
from GMA.core.utils.utils import InputPadder

from ..utils.helpers import log
from ..utils.transforms import transform, matrix2euler
from ..utils.arguments import Arguments
from ..utils.gma_parameters import GMA_Parameters
from ..utils.normalizations import get_rgb_norm
from ..odometry.network import ATDNVO
from ..localization.network import MappingVAE
from ..localization.datasets import ColorDataset

from .frame import Frame


class NeuralSLAM():
    """
    The NeuralSLAM class is the implementation of the Deep Neural SLAM architecture
    
    :param args: Arguments including SLAM output path
    :param odometry_weights: Path to odometry model weights file
    :param start_mode: Startup state for the SLAM. None means cold start which starts with empty map and odometry
    """

    def __init__(
        self, 
        args : Arguments, 
        odometry_weights : str = None, 
        start_mode : str = None
    ) -> None:

        # General arguments object, keyframe saving path and SLAM mode
        self.__gma_parameters = GMA_Parameters()
        self.__args = args
        self.__keyframes_base_path = args.keyframes_path

        self.__norm_rgb = get_rgb_norm()
        
        # Creating model and loading weights for optical flow network
        self.__flow_net = torch.nn.DataParallel(RAFTGMA(self.__gma_parameters), device_ids=[0])
        self.__flow_net.load_state_dict(torch.load(self.__gma_parameters.model))
        self.__flow_net.eval()
        self.__padder = InputPadder((3, 376, 1232))
        
        # Creating model and loading weights for odometry estimator
        self.__odometry_net = ATDNVO().to(self.__args.device)
        self.__odometry_net.load_state_dict(torch.load(odometry_weights, map_location=self.__args.device))
        self.__odometry_net.eval()
        self.__image_buffer = None

        # Property for mapping net
        self.__mapping_net = None

        # Propetries for odometry propagation
        self.__keyframes = []
        self.__last_keyframe = None
        self.__transform_propagation_matrix = torch.eye(4, dtype=torch.float32)
        self.__current_pose = torch.eye(4, dtype=torch.float32)

        # Keyframe registration parameters
        rot_threshold_deg = 10
        self.__rotation_threshold = (rot_threshold_deg/180)*torch.pi
        self.__translation_threshold = 15

        # Startup for relocalization only
        if start_mode == "mapping":
            homogenous = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 4)
            poses = torch.load(self.__args.keyframes_path + "/poses.pth")
            poses = torch.cat([poses.view(len(poses), 3, 4), homogenous.repeat(len(poses), 1, 1)], dim=1)

            files = sorted(glob.glob(self.__args.keyframes_path + "/rgb/*"))
            log("Loading keyframes")
            for i in trange(len(files)):
                self.__keyframes.append(Frame(files[i], poses[i]))
            self.__mode = "odometry"
            self.end_odometry()

        elif start_mode == "relocalization":
            self.__mapping_net = MappingVAE().to(self.__args.device).eval()
            self.__mapping_net.load_state_dict(torch.load(self.__keyframes_base_path + "/MappingVAE_weights.pth"))
            
            homogenous = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 4)
            poses = torch.load(self.__args.keyframes_path + "/poses.pth")
            poses = torch.cat([poses.view(len(poses), 3, 4), homogenous.repeat(len(poses), 1, 1)], dim=1)
            
            files = sorted(glob.glob(self.__args.keyframes_path + "/rgb/*"))
            
            log("Loading keyframes")
            for i in trange(len(files)):
                rgb = torch.load(files[i]).to(self.__args.device).float()
                if len(rgb.size()) == 3:
                    rgb = rgb.unsqueeze(0)
                mu, logvar, latent, im_pred = self.__mapping_net(rgb)
                self.__keyframes.append(Frame(files[i], poses[i], mu.detach()))
            self.__mode = "relocalization"
            
        else:
            if not os.path.exists(self.__args.keyframes_path):
                os.mkdir(self.__args.keyframes_path)
            if not os.path.exists(os.path.join(self.__args.keyframes_path, "rgb")):
                os.mkdir(os.path.join(self.__args.keyframes_path, "rgb"))
            else:
                files = glob.glob(self.__args.keyframes_path + "/rgb/*")
                
                for file in files:
                    os.remove(file)
                print("Removed ", len(files))
                
                poses_path = self.__keyframes_base_path + "/poses.pth"
                if os.path.exists(poses_path):
                    os.remove(poses_path)
                    print('Poses removed')

            self.__mode = "idle"


    def start_odometry(self):
        """
        Start odometry estimation process
        """

        # Checking if current mode is idle
        if self.__mode == "idle":
            self.__mode = "odometry"
            log("Starting odometry, accepting input image pairs")
        else:
            log("Odometry cannot be performed in current SLAM stage")


    def end_odometry(self):
        """
        Finish odometry and start the mapping process. 
        If the mapping is successful change to relocalization mode.
        """

        # Checking if SLAM mode is odometry and if there are any registered keyframes
        if (self.__mode == "odometry") and (len(self.__keyframes) > 0):
            poses = []
            for i in range(len(self.__keyframes)):
                poses.append(self.__keyframes[i].pose.flatten()[:12])
            poses = torch.stack(poses, dim=0)
            torch.save(poses, self.__keyframes_base_path + "/poses.pth")
            
            log("Odometry ended, starting mapping process...")
            # Changing mode to mapping
            self.__mode = "mapping"

            # Generating deep learning based general map
            self.__create_map()

            # Setting code for all keyframes
            with torch.no_grad():
                for frame in self.__keyframes:
                    rgb = torch.load(frame.rgb_file_name).to(self.__args.device)
                    if len(rgb.size()) == 3:
                        rgb = rgb.unsqueeze(0)
                    mu, logvar, latent, im_pred = self.__mapping_net(rgb.float())
                    frame.embedding = mu

            log("Mapping finished, changing to relocalization mode.")
            # Changin mode to relocalization
            self.__mode = "relocalization"
        elif len(self.__keyframes) == 0:
            log("There is no explored enviromnent yet!")
        else:
            log("Current state is not odometry")


    def __call__(self, im):
        """
        Call method of the NeuralSLAM class.
        It can be used in odometry and relocalization modes.
        In odometry mode input arguments are the current pair of images.
        In relocalization mode the input argument is the image to estimate camera pose from.

        :param args: The argument of the SLAM depening on the actual state detemined by mode.
        :return: In odometry mode returns the actual odometry estimation. In relocalization mode the closest relocalization estimation is returned.
        """

        with torch.no_grad():
            if self.__mode == "odometry":
                # Call functionality in odometry mode
                if self.__image_buffer is not None:    
                    # Preprocessing images
                    im1 = self.__image_buffer
                    im2 = im.to(self.__args.device)
                    im2 = TF.resize(im2, (376, 1232))
                    im2 = self.__padder.pad(im2)[0]

                    # Estimating odometry
                    _, flow = self.__flow_net(im1.unsqueeze(0), im2.unsqueeze(0), iters=12, test_mode=True)                
                    pred_rot, pred_tr = self.__odometry_net(flow)
                    pred_mat = transform(pred_rot.squeeze(), pred_tr.squeeze()).to("cpu")
                    
                    # Calculating current pose with previous actual pose and new odometry estimation
                    self.__current_pose = self.__current_pose @ pred_mat

                    # Making decision of new keyframe
                    # If keyframe decision is True, registering im2 as new keyframe
                    if self.__decide_keyframe(pred_mat):
                        index_str = str(len(self.__keyframes))
                        rgb_file_name = self.__keyframes_base_path + '/rgb/' + ('0'*(6-len(index_str))) + index_str + ".pth"
                        torch.save(im2.to("cpu").byte(),  rgb_file_name)
                        self.__keyframes.append(Frame(rgb_file_name, self.__current_pose))
                    
                    self.__image_buffer = im2
                else:
                    im = im.to(self.__args.device)
                    im = TF.resize(im, (376, 1232))
                    self.__image_buffer = self.__padder.pad(im)[0]
                    
                    rgb_file_name = self.__keyframes_base_path + "/rgb/000000.pth"
                    torch.save(im.to("cpu").byte(),  rgb_file_name)
                    self.__keyframes.append(Frame(rgb_file_name, self.__current_pose))
                
                return self.__current_pose

            elif self.__mode == "relocalization":
                # Call functionality in relocalization mode
                im_to_search = im.to(self.__args.device).float()
                if len(im_to_search.size()) == 3:
                    im_to_search = im_to_search.unsqueeze(0)
                closest_keyframe = self.__relocalize_from_image(im_to_search)
                return closest_keyframe
            
            else:
                raise Exception("SLAM called in invalid state!")


    def mode(self):
        """
        :return: Actual state of the SLAM state-machine.
        """
        return copy.deepcopy(self.__mode)


    def to(self, device):
        """
        Change which device to use with odometry and mapping networks
        """

        self.__args.device = device
        
        if self.__mode == "odometry":
            self.__flow_net = self.__flow_net.to(device)
            self.__odometry_net = self.__odometry_net.to(device)
        elif self.__mode == "relocalization":
            self.__mapping_net = self.__mapping_net.to(device)


    def get_keyframe(self, index):
        """
        Get the keyframe of the given index.
        
        :param index: Sequential id number of the keyframe
        :return: The indexed keyframe object
        """
        return self.__keyframes[index]


    def __getitem__(self, index):
        """
        Indexing operator for functionality of get_keyframe_
        """
        return self.__keyframes[index]


    def __len__(self):
        """
        Implementation of Python len() function.

        :return: The number of keyframes distinguished during odometry.
        """
        return len(self.__keyframes)


    def __decide_keyframe(self, pred_mat) -> bool:
        """
        Decide if current frame is to be registered as a keyframe.
        """
        result = False

        self.__transform_propagation_matrix = self.__transform_propagation_matrix @ pred_mat
        rotation = matrix2euler(self.__transform_propagation_matrix[:3, :3])
        translation = self.__transform_propagation_matrix[:3, -1]

        if (torch.norm(rotation) > self.__rotation_threshold) or (torch.norm(translation) > self.__translation_threshold):
            result = True
            self.__transform_propagation_matrix = torch.eye(4, dtype=torch.float32)

        return result


    def __create_map(self):
        """
        Generate autoencoder based latent space map
        """

        num_epochs = 5 # TODO increase!
        batch_size = 8
        running_losses = []

        # Creating model for mapping net
        self.__mapping_net = MappingVAE().to(self.__args.device).train()

        dataset = ColorDataset(self.__keyframes_base_path, pth=True)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(self.__mapping_net.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs*len(dataloader), eta_min=1e-5)

        #aug = transforms.Compose([
        #    transforms.ColorJitter(brightness=0.1, saturation=0.1, hue=1e-4),
        #])
        for i in trange(num_epochs):
            running_loss = 0
            for batch, im in enumerate(dataloader):
                optimizer.zero_grad()
                im = im.to(self.__args.device)

                mu, logvar, latent, im_pred = self.__mapping_net(im)

                im = TF.resize(im, list(im_pred.size()[-2:]))
                im = TF.gaussian_blur(im, [5, 5])
                im = self.__norm_rgb(im)
                
                loss1 = ((im_pred - im)**2).mean()
                sat_true = (im.amax(dim=1) - im.amin(dim=1))
                sat_pred = (im_pred.amax(dim=1) - im_pred.amin(dim=1))
                loss2 = (sat_true - sat_pred).abs().mean()
                loss = loss1 + loss2
                running_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
            
            model_name = self.__keyframes_base_path + "/MappingVAE_weights.pth"
            torch.save(self.__mapping_net.state_dict(), model_name)
            running_losses.append(running_loss/len(dataloader))
        
        self.__mapping_net.eval()
        torch.save(torch.tensor(running_losses), "mapping_loss.pth")


    def __relocalize_from_image(self, image):
        """
        Relocalization method
        """

        mu, logvar, latent, im_pred = self.__mapping_net(image)

        closest_keyframe, distances = self.__get_closest_keyframe(mu)

        initial_pose = closest_keyframe.pose
        
        pose_diff = self.__refine_localization(closest_keyframe, image)

        refined_pose = initial_pose @ pose_diff

        return initial_pose, refined_pose, distances


    def __get_closest_keyframe(self, code):
        """
        Searching for the keyframe with the closest code to the one given as argument
        """
        distances = []
        for i in range(len(self.__keyframes)):
            distances.append(torch.norm((self.__keyframes[i].embedding-code), p=2))

        distances = torch.stack(distances, dim=0)
        pred_index = torch.argmin(distances)

        return self.__keyframes[pred_index], distances


    def __refine_localization(self, closest_keyframe,  im_to_search):
        """
        Refining current relocalization estimate
        """
        closest_rgb = torch.load(closest_keyframe.rgb_file_name).unsqueeze(0)
        im1, im2 = closest_rgb.to(self.__args.device), im_to_search.to(self.__args.device)

        # Calculating optical flow values
        _, flow = self.__flow_net(im1, im2, iters=12, test_mode=True) # TODO check unsqueeze
        pred_rot, pred_tr = self.__odometry_net(flow)
        pred_mat = transform(pred_rot.squeeze(), pred_tr.squeeze()).to("cpu")

        return pred_mat

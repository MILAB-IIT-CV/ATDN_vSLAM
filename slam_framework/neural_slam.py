from tokenize import String
from xmlrpc.client import Boolean
import torch
from GMA.core.network import RAFTGMA
from helpers import GMA_Parameters, Arguments, log, transform, matrix2euler
from odometry.clvo import CLVO
from localization.localization import MappingVAE
from slam_framework.frame import Frame
import os
import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from localization.localization_dataset import MappingDataset
import copy



class NeuralSLAM():
    """
    The NeuralSLAM class is the implementation of the Deep Neural SLAM architecture
    """

    def __init__(self, args : Arguments, start_mode: String = None) -> None:
        """
        keyframes_base_path: The path to save keyframes to, given as a string.
        """

        # General arguments object, keyframe saving path and SLAM mode
        self.__gma_parameters = GMA_Parameters()
        self.__args = args
        self.__keyframes_base_path = args.keyframes_path # TODO checking existance

        # Normalization parameters
        self.__rgb_mean = torch.load("normalization_cache/rgb_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(self.__args.device)
        self.__rgb_std = torch.load("normalization_cache/rgb_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(self.__args.device)
        self.__flows_mean = torch.load("normalization_cache/flow_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(self.__args.device)
        self.__flows_std = torch.load("normalization_cache/flow_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(self.__args.device)
        
        # Creating model and loading weights for optical flow network
        self.__flow_net = torch.nn.DataParallel(RAFTGMA(self.__gma_parameters), device_ids=[0])
        self.__flow_net.load_state_dict(torch.load(self.__gma_parameters.model))
        self.__flow_net.eval()
        
        # Creating model and loading weights for odometry estimator
        self.__odometry_net = CLVO(args=self.__gma_parameters).to(self.__args.device)
        self.__odometry_net.load_state_dict(torch.load("odometry/clvo_final_adam_3.pth", map_location=self.__args.device))
        self.__odometry_net.eval()

        # Property for mapping net
        self.__mapping_net = None

        # Propetries for odometry propagation
        self.__keyframes = []
        self.__last_keyframe = None
        self.__transform_propagation_matrix = torch.eye(4, dtype=torch.float32)
        self.__translation_propagation_vector = torch.zeros((3))
        self.__current_pose = torch.eye(4, dtype=torch.float32)

        # Keyframe registration parameters
        rot_threshold_deg = 10
        self.__rotation_threshold = (rot_threshold_deg/180)*torch.pi
        self.__translation_threshold = 15

        self.__precomputed_flow = args.precomputed_flow

        # Startup for relocalization only
        if start_mode == "mapping":
            poses = torch.load(self.__args.keyframes_path + "/poses.pth")
            files = sorted(glob.glob(self.__args.keyframes_path + "/rgb/*"))
            log("Loading keyframes")
            for i in range(len(files)):
                self.__keyframes.append(Frame(files[i], 
                                              torch.stack([poses[i][:4], 
                                                           poses[i][4:8], 
                                                           poses[i][8:], 
                                                           torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)])
                                            )
                                        )
                print('    ', end='\r')
                print(i, end='')
            print()
            self.__mode = "odometry"
            self.end_odometry()

        elif start_mode == "relocalization":
            self.__mapping_net = MappingVAE().to(self.__args.device).eval()
            self.__mapping_net.load_state_dict(torch.load(self.__keyframes_base_path + "/MappingVAE_weights.pth"))
            poses = torch.load(self.__args.keyframes_path + "/poses.pth")
            files = sorted(glob.glob(self.__args.keyframes_path + "/rgb/*"))
            
            log("Loading keyframes")
            for i in range(len(files)):
                rgb = torch.load(files[i]).to(self.__args.device)
                mu, logvar, latent, im_pred = self.__mapping_net(rgb, VAE=True)

                self.__keyframes.append(Frame(files[i], 
                                               torch.stack([poses[i][:4], 
                                                          poses[i][4:8], 
                                                          poses[i][8:], 
                                                          torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)]),
                                               mu.detach()
                                            )
                                        )
                print('    ', end='\r')
                print(i, end='')
            print()
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
        This method is used to start odometry.
        """

        # Checking if current mode is idle
        if self.__mode == "idle":
            # Changing SLAM mode to odometry
            self.__mode = "odometry"
            log("Starting odometry, accepting input image pairs")
        else:
            log("Odometry cannot be performed in current SLAM stage")


    def end_odometry(self):
        """
        This method is used to end odometry mode, start the mapping process, and after the successful mapping changing to relocalization mode.
        """

        # Checking if SLAM mode is odometry and if there is any registered keyframes
        if (self.__mode == "odometry") and (len(self.__keyframes) > 0):
            # Freeing memory of odometry network parts

            poses = []
            for i in range(len(self.__keyframes)):
                pose = self.__keyframes[i].pose
                pose = torch.cat([pose[0, :], pose[1, :], pose[2, :]], dim=0)
                poses.append(pose)
            poses = torch.stack(poses, dim=0)
            torch.save(poses, self.__keyframes_base_path + "/poses.pth")

            
            log("Odometry ended, starting mapping process...")
            # Changing mode to mapping
            self.__mode = "mapping"

            # Generating deep learning based general map
            self.__create_map()

            with torch.no_grad():
                for frame in self.__keyframes:
                    rgb = torch.load(frame.rgb_file_name).to(self.__args.device)
                    mu, logvar, latent, im_pred = self.__mapping_net(rgb, VAE=True)
                    frame.embedding = mu

            log("Mapping finished, changing to relocalization mode.")
            # Changin mode to relocalization
            self.__mode = "relocalization"
        else:
            log("There is no explored enviromnent yet!")


    def __call__(self, *args):
        """
        Call method of the NeuralSLAM class.
        It can be used in odometry and relocalization modes.
        In odometry mode input arguments are the current pair of images.
        In relocalization mode the input argument is the image to estimate camera pose from.
        """

        if self.__mode == "odometry":
        # Call functionality in odometry mode
            # Extracting images from input arguments
            with torch.no_grad():
                if not self.__precomputed_flow:
                    im1, im2 = args[0], args[1]
                    # Preprocessing of images: unused dimension reduction, transfer to processing device, padding
                    im1, im2 = im1.to(self.__args.device), im2.to(self.__args.device)

                    # Calculating optical flow values
                    _, flow_up = self.__flow_net(im1.unsqueeze(0), im2.unsqueeze(0), iters=12, test_mode=True)
                else:
                    im1, im2, flow_up = args[0].to(self.__args.device), args[1].to(self.__args.device), args[2].to(self.__args.device)
                
                # Preparing input data for odometry estimator network
                im1 = (im1-self.__rgb_mean)/self.__rgb_std
                im2 = (im2-self.__rgb_mean)/self.__rgb_std
                flow = (flow_up-self.__flows_mean)/self.__flows_std
                input_data_batch = torch.cat([flow, im1, im2], dim=1) # TODO check ig padded or not padded is needed
                
                # Estimating odometry
                odometry_result = []
                for input_data in input_data_batch:
                    pred_rot, pred_tr = self.__odometry_net(input_data.unsqueeze(0))

                    # Convert euler angles rotation and translation vector to 4*4 transform matrix
                    pred_mat = transform(pred_rot, pred_tr)
                    
                    # Calculating current pose with previous actual pose and new odometry estimation
                    self.__current_pose = torch.matmul(self.__current_pose, pred_mat)
                    odometry_result.append(self.__current_pose)

                    # Making decision of new keyframe
                    # If keyframe decision is True, registering im2 as new keyframe
                    is_new_keyframe = self.__decide_keyframe(pred_mat)
                    if is_new_keyframe:
                        index_str = str(len(self.__keyframes))
                        rgb_file_name = self.__keyframes_base_path + '/rgb/' + ('0'*(6-len(index_str))) + index_str + ".pth"
                        torch.save(im2.detach().cpu(),  rgb_file_name)
                        self.__keyframes.append(Frame(rgb_file_name, self.__current_pose))
                return odometry_result

        elif self.__mode == "relocalization":
        # Call functionality in relocalization mode
            im_to_search = args[0].to(self.__args.device)
           
            closest_keyframe = self.__relocalize_from_image(im_to_search)
            return closest_keyframe
            #refined_pose = self.__refine_localization(closest_keyframe,  im_to_search)
            #return refined_pose


    def mode(self):
        """
        Getter for current SLAM mode
        """

        return copy.deepcopy(self.__mode)


    def to(self, device):
        """
        Method change device to load deep learning models on
        """

        self.__args.device = device
        
        if self.__mode == "odometry":
            self.__flow_net = self.__flow_net.to(device)
            self.__odometry_net = self.__odometry_net.to(device)
        elif self.__mode == "relocalization":
            self.__mapping_net = self.__mapping_net.to(device)


    def get_keyframe(self, index):
        return self.__keyframes[index]

    def __getitem__(self, index):
        return self.__keyframes[index]

    def __len__(self):
        return len(self.__keyframes)

    def __decide_keyframe(self, pred_mat) -> Boolean:
        """
        Method to decide if current frame is to be registered as a keyframe
        """
        result = False

        self.__transform_propagation_matrix = torch.matmul(self.__transform_propagation_matrix, pred_mat)
        self.__translation_propagation_vector += pred_mat[:-1, -1]
        r = matrix2euler(self.__transform_propagation_matrix[:3, :3])
        if (torch.norm(r) > self.__rotation_threshold) or (torch.norm(self.__translation_propagation_vector) > self.__translation_threshold):
            result = True
            self.__transform_propagation_matrix = torch.eye(4, dtype=torch.float32)
            self.__translation_propagation_vector = torch.zeros((3))

        return result


    def __create_map(self):
        """
        Generating autoencoder based general map
        """

        # TODO implement training loop for mapping net
        num_epochs = 10
        batch_size = 8

        dataset = MappingDataset(self.__keyframes_base_path, slam=True)

        # Creating model for mapping net
        target_shape = (dataset[0].shape[-2], dataset[0].shape[-1])
        self.__mapping_net = MappingVAE(target_size=target_shape).to(self.__args.device).train()


        rgb_mean = self.__rgb_mean
        rgb_sigma = self.__rgb_std

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.__mapping_net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs*len(dataloader), eta_min=1e-5)

        #aug = transforms.Compose([
        #    transforms.ColorJitter(brightness=0.1, saturation=0.1, hue=1e-4),
        #])

        for i in range(num_epochs):
            print("-------------------- Epoch", i+1, "/", num_epochs, " --------------------")
            
            for batch, im in enumerate(dataloader):
                
                optimizer.zero_grad()
                im = im.to(self.__args.device)
                im_normalized = im

                #im_normalized = (im-rgb_mean)/rgb_sigma
                #im_normalized = ((aug(im.byte())-rgb_mean)/rgb_sigma).float()

                mu, logvar, latent, im_pred = self.__mapping_net(im_normalized, VAE=False)

                loss = F.mse_loss(im_pred, im_normalized)
                #loss2 = torch.abs(torch.norm(torch.exp(0.5*logvar), p=2)-1.0)
                #loss = loss1
                #loss = loss1 + loss2

                loss.backward()
                optimizer.step()
                scheduler.step()

                print("Iteration: ", batch, "/", len(dataloader), "\t\t Loss: ", loss.item(), "\t LR: ", scheduler.get_last_lr())
            
            model_name = self.__keyframes_base_path + "/MappingVAE_weights.pth"
            log("Saving model as ", model_name)
            torch.save(self.__mapping_net.state_dict(), model_name)
        self.__mapping_net = self.__mapping_net.eval()



    def __relocalize_from_image(self, image):
        """
        Relocalization method
        """

        # TODO get closest keyframe
        mu, logvar, latent, im_pred = self.__mapping_net(image, VAE=True)

        distances = []
        for i in range(len(self.__keyframes)):
            distances.append(torch.norm((self.__keyframes[i].embedding-mu), p=2))


        distances = torch.stack(distances, dim=0)
        pred_index = torch.argmin(distances)

        initial_pose = self.__keyframes[pred_index].pose

        pose_diff = self.__refine_localization(self.__keyframes[pred_index], image)

        refined_pose = torch.matmul(initial_pose, pose_diff)

        return initial_pose, refined_pose, distances


    def __refine_localization(self, closest_keyframe,  im_to_search):
        """
        Refining current relocalization estimate
        """

        with torch.no_grad():
            closest_rgb = torch.load(closest_keyframe.rgb_file_name)

            im1, im2 = closest_rgb.to(self.__args.device), im_to_search.to(self.__args.device)

            # Calculating optical flow values
            _, flow_up = self.__flow_net(im1, im2, iters=12, test_mode=True)
            
            # Preparing input data for odometry estimator network
            im1 = (im1-self.__rgb_mean)/self.__rgb_std
            im2 = (im2-self.__rgb_mean)/self.__rgb_std
            flow = (flow_up-self.__flows_mean)/self.__flows_std
            input_data = torch.cat([flow, im1, im2], dim=1)
            
            # Estimating odometry
            pred_rot, pred_tr = self.__odometry_net(input_data)
            pred_mat = transform(pred_rot, pred_tr)

            return pred_mat

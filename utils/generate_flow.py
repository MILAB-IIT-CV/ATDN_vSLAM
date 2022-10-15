import os
import torch
from GMA.core.network import RAFTGMA
from utils.gma_parameters import GMA_Parameters
from utils.arguments import Arguments
from utils.helpers import log
from odometry.vo_datasets import KittiOdometryDataset

gma_parameters = GMA_Parameters()
args = Arguments.get_arguments()

gma = torch.nn.DataParallel(RAFTGMA(gma_parameters), device_ids=[0])
gma.load_state_dict(torch.load(gma_parameters.model))

with torch.no_grad():
    sequences = sorted(os.listdir(args.data_path+"/dataset/sequences"))
    for sequence in sequences:
        dataset = KittiOdometryDataset(args.data_path, 
                                             sequence, 
                                             precomputed_flow=False, 
                                             sequence_length=1)
        log("Generating sequence No. " + sequence + " with " + str(len(dataset)) + " data")
        
        directory = args.data_path+"/dataset/flows/"+sequence
        if(not os.path.exists(directory)):
            os.mkdir(directory)
        
        for i in range(len(dataset)):
            # Preprocessing of images: unused dimension reduction, transfer to processing device, padding
            ims, rot, tr = dataset[i]
            im1, im2 = ims[0].unsqueeze(0).to(args.device), ims[1].unsqueeze(0).to(args.device)
            
            # Calculating optical flow values
            _, flow_up = gma(im1, im2, iters=12, test_mode=True)
            
            flow_file = (6-len(str(i)))*'0'+str(i) + ".pt"        
            torch.save(flow_up, args.data_path+"/dataset/flows/"+sequence+"/"+flow_file)

            print('    ', end='\r')
            print(i, end='')
        print()

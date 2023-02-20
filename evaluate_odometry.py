import os
import argparse
import time

# PyTorch imports
import torch

# Other external package imports
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

# Project module imports
from odometry.datasets import FlowKittiDataset2
from odometry.network import ATDNVO
from utils.helpers import log
from utils.transforms import transform, rel2abs
from utils.arguments import Arguments


def run_inference(model, args, sequence, exp, stage, forward):
    """
    Inference runner on a chosen KITTI odometry sequence with a specified learning stage.
    It is also changeable, wheter to run the inference forward or backward.
    :param sequence: The choesen KITTI sequence to run inference on
    :param stage: The chosen learning stage weight to use with the model
    :param forward: Integer which controls wheter to run inference forward or backward. >=0.5 means forward <-1 mean backward. 
    In between numbers are causing uniformly random sampled forward and backward.
    """

    use_model = True
    sequence_length = 1

    # Instantiating dataset and dataloader
    dataset = FlowKittiDataset2(args.data_path, [sequence], augment=forward, sequence_length=sequence_length)

    model_code = str(model.__class__.__name__) + model.suffix
    load_file = "checkpoints/" + str(exp) + '_' + str(stage) + model_code.lower() + ".pth"
    log("Load file:", load_file)
    model.load_state_dict(torch.load(load_file, map_location=args.device))
    
    eval_time = 0.0
    with torch.no_grad():
        model.to(args.device)
        model.eval()

        translations = []
        rotations = []

        count = 0
        if forward == 1:
            start = 0
            end = len(dataset)
            step = 1*sequence_length
        else:
            start = len(dataset)-1
            end = -1
            step = -1*sequence_length
        

        for i in trange(start, end, step):
            flow, rot, tr = dataset[i]
            if use_model:
                flow = flow.unsqueeze(0).to(args.device).float()

                start_time = time.time()
                rot, tr = model(flow)
                end_time = time.time()
                eval_time += end_time-start_time
            rot = rot.to("cpu")
            tr = tr.to("cpu")
            
            rotations.append(rot)        
            translations.append(tr)
            
            count = count+1

        log("Inference Done! Accumulated time of inference in seconds: ", eval_time)
        #log("Rot length: ", len(rotations))
        #log("Tr length: ", len(translations))

    return rotations, translations


def save_results(abs_poses, exp, arch, stage, seq, forward, need_plot=True):
    # Plotting results
    numpy_poses = []
    suffix = "f" if forward>0 else "b"
    for i in range(len(abs_poses)):
        numpy_poses.append((np.array(abs_poses[i][:3, :].view(12).numpy())))
    numpy_poses = np.stack(numpy_poses, axis=0)

    exp_path = os.path.join("eval", "results", str(exp))
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    exp_path = os.path.join(exp_path, arch)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
        
    np.savetxt(exp_path + '/' + str(exp) + '_' + arch + "_" + seq + "_" + suffix + ".txt", numpy_poses)

    X = np.array([p[3] for p in numpy_poses])
    Z = np.array([p[-1] for p in numpy_poses])

    if need_plot:
        plt.plot(X, Z)
        plt.savefig("s_" + seq + "_exp_" + str(exp) + "_stage_" + str(stage) + "_" + suffix + ".png")



def main():
    
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--stage", type=int)
    parser.add_argument("--sequence", type=str, default='00')
    parser.add_argument("--exp", type=int, default=6)
    parser.add_argument("--forward", type=int)
    parser.add_argument("--plot", type=bool, default=True)
    args = parser.parse_args()
    
    if args.stage is None or args.forward is None:
        raise Exception("Error! No stage given!")

    model_args = Arguments.get_arguments()
    model = ATDNVO().to(model_args.device)
    arch = str(model.__class__.__name__) + model.suffix

    # Forward = 0 means both forward and backward run
    if args.forward == 0:
        # Forward
        rotations, translations = run_inference(model, model_args, args.sequence, args.exp, args.stage, 1)
        abs_poses = rel2abs(rotations, translations)
        save_results(abs_poses, args.exp, arch, args.stage, args.sequence, 1, False)

        # Backward
        rotations, translations = run_inference(model, model_args, args.sequence, args.exp, args.stage, -1)
        abs_poses = rel2abs(rotations, translations)
        save_results(abs_poses, args.exp, arch, args.stage, args.sequence, -1, False)
    else:
        rotations, translations = run_inference(model, model_args, args.sequence, args.exp, args.stage, args.forward)
        abs_poses = rel2abs(rotations, translations)
        save_results(abs_poses, args.exp, arch, args.stage, args.sequence, args.forward, args.plot)

main()
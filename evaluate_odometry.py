import argparse
import time

# PyTorch imports
import torch

# Other external package imports
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

# Project module imports
from odometry.datasets import FlowKittiDataset
from odometry.clvo import CLVO
from utils.helpers import log, transform
from utils.arguments import Arguments


def run_inference(sequence, exp, stage, forward):
    """
    Inference runner on a chosen KITTI odometry sequence with a specified learning stage.
    It is also changeable, wheter to run the inference forward or backward.
    :param sequence: The choesen KITTI sequence to run inference on
    :param stage: The chosen learning stage weight to use with the model
    :param forward: Integer which controls wheter to run inference forward or backward. >=0.5 means forward <-1 mean backward. 
    In between numbers are causing uniformly random sampled forward and backward.
    """
    # Instantiating arguments object for optical flow module
    args = Arguments.get_arguments()
    use_model = True
    sequence_length = 1

    # Instantiating dataset and dataloader
    dataset = FlowKittiDataset(args.data_path, [sequence], augment=forward, sequence_length=sequence_length)

    model = CLVO().to(args.device)
    model.load_state_dict(torch.load("checkpoints/clvo_generalization"+str(exp)+"_" + str(stage) + ".pth", map_location=args.device))
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
                flow = flow.unsqueeze(0).to(args.device)

                start_time = time.time()
                rot, tr = model(flow)
                end_time = time.time()
                eval_time += end_time-start_time
            rot = rot.detach()
            tr = tr.detach()
            
            rotations.append(rot)        
            translations.append(tr)
            
            count = count+1

        log("Inference Done! Accumulated time of inference in seconds: ", eval_time)
        #log("Rot length: ", len(rotations))
        #log("Tr length: ", len(translations))

    return rotations, translations


def rel2abs(rotations, translations):
    """Relative to absolute system conversion
    rotations: list of euler angle rotations
    translations: list of translation vectors
    returns: M*4*4 tensor containing absolute pose matrixes
    """
    homogenous = []

    instance_num = len(rotations)
    for i in range(instance_num):
        homogenous.append(transform(rotations[i].squeeze(), translations[i].squeeze()))

    global_scale = [torch.eye(4, dtype=torch.float32)]
    for i in range(0, instance_num):
        global_scale.append(torch.matmul(global_scale[i], homogenous[i]))
        
    abs_poses = torch.stack(global_scale, dim=0)

    return abs_poses


def save_results(abs_poses, exp, stage, seq, forward, need_plot=True):
    # Plotting results
    numpy_poses = []
    suffix = "f" if forward>0 else "b"
    for i in range(len(abs_poses)):
        numpy_poses.append((np.array(abs_poses[i][:3, :].view(12).numpy())))
    numpy_poses = np.stack(numpy_poses, axis=0)

    np.savetxt("eval/results"+seq+"_" + str(exp) + "_" + str(stage) + "_" + suffix + ".txt", numpy_poses)

    X = np.array([p[3] for p in numpy_poses])
    Z = np.array([p[-1] for p in numpy_poses])

    if need_plot:
        plt.plot(X, Z)
        plt.savefig("s_"+seq+"_exp_"+str(exp)+"_stage_" + str(stage) + "_" + suffix + ".png")



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

    # Forward = 0 means both forward and backward run
    if args.forward == 0:
        # Forward
        rotations, translations = run_inference(args.sequence, args.exp, args.stage, 1)
        abs_poses = rel2abs(rotations, translations)
        save_results(abs_poses, args.exp, args.stage, args.sequence, 1, False)

        # Backward
        rotations, translations = run_inference(args.sequence, args.exp, args.stage, -1)
        abs_poses = rel2abs(rotations, translations)
        save_results(abs_poses, args.exp, args.stage, args.sequence, -1, False)
    else:
        rotations, translations = run_inference(args.sequence, args.exp, args.stage, args.forward)
        abs_poses = rel2abs(rotations, translations)
        save_results(abs_poses, args.exp, args.stage, args.sequence, args.forward, args.plot)

main()
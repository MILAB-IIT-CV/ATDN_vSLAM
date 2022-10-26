# PyTorch imports
import torch

# Other external package imports
import matplotlib.pyplot as plt
import numpy as np

# Project module imports
from odometry.vo_datasets import FlowKittiDataset
from odometry.clvo import CLVO
from utils.helpers import log, transform
from utils.arguments import Arguments
from utils.normalization import FlowStandardization


def run_inference(sequence, stage):
    # Instantiating arguments object for optical flow module
    args = Arguments.get_arguments()

    # Instantiating dataset and dataloader
    batch_size = 16
    sequence_length = 1
    dataset = FlowKittiDataset(args.data_path, [sequence], augment=False, sequence_length=1)

    model = CLVO().to(args.device)
    model.load_state_dict(torch.load("checkpoints/clvo_generalization5_" + str(stage) + ".pth", map_location=args.device))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Trainable parameters:", trainable_params)

    normalization =  FlowStandardization().eval()

    with torch.no_grad():
        model.to(args.device)
        model.eval()

        translations = []
        rotations = []
        #resize = Resize((376, 1248))
        use_model = True

        count = 0
        for i in range(0, len(dataset), 1):
            flow, rot, tr = dataset[i]
            if use_model:
                flow = normalization(flow.unsqueeze(0).to(args.device))

                rot, tr = model(flow)
            rot = rot.detach()
            tr = tr.detach()
            
            rotations.append(rot)        
            translations.append(tr)
            
            #for j in range(len(rot)):
            #    translation = tr[j].squeeze()
            #    rotation = rot[j].squeeze()

                #translations.append(translation)
                #rotations.append(rotation)

            print('    ', end='\r')
            print(count, end='')
            count = count+1

        print()
        log("Data loading Done!")
        log("Rot length: ", len(rotations))
        log("Tr length: ", len(translations))

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

    global_scale = []
    global_scale.append(homogenous[0])
    for i in range(1, instance_num):
        global_scale.append(torch.matmul(global_scale[i-1], homogenous[i]))
        
    abs_poses = torch.stack(global_scale, dim=0)

    return abs_poses


def save_results(abs_poses, stage):
    # Plotting results
    numpy_poses = []

    for i in range(len(abs_poses)):
        numpy_poses.append((np.array(abs_poses[i][:3, :].view(12).numpy())))
    numpy_poses = np.stack(numpy_poses, axis=0)

    np.savetxt("results.txt", numpy_poses)
    reloaded_poses = np.loadtxt("results.txt")

    X = np.array([p[3] for p in reloaded_poses])
    Z = np.array([p[-1] for p in reloaded_poses])

    plt.plot(X, Z)
    plt.savefig("stage_" + str(stage) + "_eval.png")



def main():
    stage = 4

    rotations, translations = run_inference('00', stage)
    abs_poses = rel2abs(rotations, translations)
    save_results(abs_poses, stage)

main()
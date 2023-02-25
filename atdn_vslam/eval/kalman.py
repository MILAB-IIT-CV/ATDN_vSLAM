import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

from ..utils.transforms import matrix2euler, rel2abs


def preprocess_poses_euler(poses):
    # Stacking the matrix rows stored in the lines of the array
    rotations = []
    translations = []
    for i in range(len(poses)-1):
        pose1 = torch.cat([poses[i], torch.tensor([[0, 0, 0, 1]])], dim=0)
        inverted1 = torch.inverse(pose1)

        pose2 = torch.cat([poses[i+1], torch.tensor([[0, 0, 0, 1]])], dim=0)
        
        delta_pose = torch.matmul(inverted1, pose2)
        delta_rot = delta_pose[:3, :3]
        delta_translation = delta_pose[:3, -1]

        delta_rotation = matrix2euler(delta_rot)

        rotations.append(delta_rotation)
        translations.append(delta_translation)

    return torch.stack(rotations), torch.stack(translations)


def plot_error(err, title, file):
    fig, axs = plt.subplots(1, 3, sharey='all')
    example = err[:, 0].numpy()
    axs[0].hist(example, np.arange(example.min(), example.max(), 0.0001))
    #plt.subplots(1, 3, 2, sharey='col')
    example = err[:, 1].numpy()
    axs[1].hist(example, np.arange(example.min(), example.max(), 0.0001))
    #plt.subplots(1, 3, 3, sharey='col')
    example = err[:, 2].numpy()
    axs[2].hist(example, np.arange(example.min(), example.max(), 0.0001))
    axs[1].set_title(title)
    plt.savefig(file)


def kalman(x1, x2, s1, s2):
    var1 = (s1**2).unsqueeze(0)
    var2 = (s2**2).unsqueeze(0)
    
    x_opt = ((x1*var2)+(x2*var1))/(var1+var2)
    return x_opt


def process_kalman(args, STD):
    std_rot_f, std_rot_b, std_tr_f, std_tr_b = STD

    dir_path = "eval/results/" + args.exp + '/' + args.model + '/'
    filename = str(args.exp) + '_' + args.model + '_' + args.sequence

    real = np.loadtxt("eval/GT/" + args.sequence + ".txt")
    forward = np.loadtxt(dir_path + filename + "_f.txt")
    backward = np.loadtxt(dir_path + filename + "_b.txt")
    
    mat_f = torch.from_numpy(forward).view(forward.shape[0], 3, 4)
    mat_b = torch.from_numpy(backward).view(forward.shape[0], 3, 4)

    #mat_b2 = [torch.cat([m, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0) for m in mat_b]
    h_extension = torch.tensor([0, 0, 0, 1], dtype=mat_b.dtype).view(1, 1, 4).repeat(mat_b.shape[0], 1, 1)
    mat_b2 = torch.cat([mat_b, h_extension], dim=1)
    
    inverse_mat = torch.inverse(mat_b2[-1])
    mat_b = torch.flip(torch.stack([torch.matmul(inverse_mat, m)[:3, :] for m in mat_b2]), dims=[0])    
    backward2 = mat_b.view(-1, 12)
    np.savetxt(dir_path + filename +"_bt.txt", backward2.numpy())

    rot_f, tr_f = preprocess_poses_euler(mat_f)
    rot_b, tr_b = preprocess_poses_euler(mat_b)

    opt_rot = kalman(rot_f, rot_b, std_rot_f, std_rot_b)
    opt_tr = kalman(tr_f, tr_b, std_tr_f, std_tr_b)

    opt_mat = rel2abs(opt_rot, opt_tr)

    np.savetxt(dir_path + filename + "_k.txt", opt_mat[:, :3, :].view((-1, 12)).numpy())

    opt_pos = opt_mat[:, :3, -1]
    plt.plot(opt_pos[:, 0], opt_pos[:, -1]); plt.title("Kalman"); plt.savefig("kalman.png")
    plt.close()
    print("Kalman Done!")


def determine_std(args):
    dir_path = "eval/results/" + args.exp + '/' + args.model + '/'
    filename = str(args.exp) + '_' + args.model + '_00'

    real = np.loadtxt("eval/GT/00.txt")
    forward = np.loadtxt(dir_path + filename + "_f.txt")
    backward = np.loadtxt(dir_path + filename + "_b.txt")
    
    mat_r = torch.from_numpy(real).view(real.shape[0], 3, 4)
    mat_f = torch.from_numpy(forward).view(forward.shape[0], 3, 4)
    mat_b = torch.from_numpy(backward).view(forward.shape[0], 3, 4)

    #mat_b2 = [torch.cat([m, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0) for m in mat_b]
    h_extension = torch.tensor([0, 0, 0, 1], dtype=mat_b.dtype).view(1, 1, 4).repeat(mat_b.shape[0], 1, 1)
    mat_b2 = torch.cat([mat_b, h_extension], dim=1)
    
    inverse_mat = torch.inverse(mat_b2[-1])
    mat_b = torch.flip(torch.stack([torch.matmul(inverse_mat, m)[:3, :] for m in mat_b2]), dims=[0])    
    backward2 = mat_b.view(-1, 12)
    np.savetxt(dir_path + filename +"_bt.txt", backward2.numpy())

    rot_f, tr_f = preprocess_poses_euler(mat_f)
    rot_b, tr_b = preprocess_poses_euler(mat_b)
    rot_r, tr_r = preprocess_poses_euler(mat_r)
    
    error_rot_f = rot_f-rot_r
    error_rot_b = rot_b-rot_r
    error_tr_f = tr_f-tr_r
    error_tr_b = tr_b-tr_r

    std_rot_f = error_rot_f.std((0))
    std_rot_b = error_rot_b.std((0))
    std_tr_f = error_tr_f.std((0))
    std_tr_b = error_tr_b.std((0))

    return [std_rot_f, std_rot_b, std_tr_f, std_tr_b]


def main():

    parser = argparse.ArgumentParser(description="Kalman filter postprocessing optimizer script")
    #parser.add_argument("--stage", type=int)
    parser.add_argument("--exp", type=str, default=8)
    parser.add_argument("--sequence", type=str, default='00')
    parser.add_argument("--model", type=str)
    
    args = parser.parse_args()

    STD = determine_std(args)
    for t in STD:
        print(t)
    process_kalman(args, STD)


main()
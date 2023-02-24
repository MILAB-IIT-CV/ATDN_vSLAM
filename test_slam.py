import os
import glob
import time

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from atdn_vslam.slam_framework.neural_slam import NeuralSLAM
from atdn_vslam.localization.datasets import ColorDataset
from atdn_vslam.odometry.datasets import KittiOdometryDataset
from atdn_vslam.utils.arguments import Arguments
from atdn_vslam.utils.transforms import matrix2euler
from atdn_vslam.utils.helpers import log


def to_vectors(mat):
    orientation = matrix2euler(mat[:3, :3])
    position = mat[:3, -1]
    return orientation, position


def test_odometry(args, weights_file):
    dataset = ColorDataset(data_path=args.data_path, sequence="00")
    slam = NeuralSLAM(args, odometry_weights=weights_file)
    
    f = open("test_results/log.txt", 'a')
    f.write("\nOdometry test \n-------------\n")

    slam.start_odometry()
    f.write("SLAM mode: " + str(slam.mode()) + '\n')

    global_scale = []
    slam_call_time = []

    for i in trange(len(dataset)):
        img = dataset[i]
        
        start = time.time()
        current_pose = slam(img.squeeze())
        end = time.time()
        
        slam_call_time.append(end-start)
        global_scale.append(current_pose)

    global_scale = torch.stack(global_scale, dim=0)
    slam_call_time = np.array(slam_call_time)

    f.write("Average odometry time: " + str(slam_call_time.mean()) + '\n')
    f.write("Odometry time std: " + str(slam_call_time.std()) + '\n')
    fps_manual = 1/(slam_call_time.mean())
    f.write("FPS from time: " + str(1/slam_call_time.mean()) + '\n')

    slam.end_odometry()
    
    DATA_PATH = args.keyframes_path + "/poses.pth"
    poses = torch.load(DATA_PATH)
    f.write(str(poses.shape) + '\n')
    X = poses[:, 3]
    Z = poses[:, -1]
    plt.figure()
    plt.scatter(X.numpy(), Z.numpy())
    plt.savefig("test_results/keyframes_from_file.png")

    keyframe_positions = []

    for i in range(len(slam)):
        pose = slam[i].pose
        keyframe_positions.append(pose[:3, 3])

    keyframe_positions = torch.stack(keyframe_positions, dim=0).to("cpu")
    global_pos = global_scale[:, :3, -1].to('cpu')
    X, Y, Z = global_pos[:, 0], global_pos[:, 1], global_pos[:, 2]
    X_key, Y_key, Z_key = keyframe_positions[:, 0], keyframe_positions[:, 1], keyframe_positions[:, 2]

    plt.figure()
    plt.plot(X, Z)
    plt.scatter(X_key, Z_key)
    plt.savefig("test_results/keyframes_from_indexing.png")

    return global_pos


def test_remapping(args, weights_file, global_pos):
    slam = NeuralSLAM(args, odometry_weights=weights_file, start_mode="mapping")
    X, Y, Z = global_pos[:, 0], global_pos[:, 1], global_pos[:, 2]

    keyframe_positions = []

    for i in range(len(slam)):
        keyframe_positions.append(slam[i].pose[:3, 3])
    
    keyframe_positions = torch.stack(keyframe_positions, dim=0).to("cpu")
    X_key, Y_key, Z_key = keyframe_positions[:, 0], keyframe_positions[:, 1], keyframe_positions[:, 2]

    plt.figure()
    plt.plot(X, Z)
    plt.scatter(X_key, Z_key)
    plt.savefig("test_results/remapping_poses.png")

def test_relocalization(args, weights_file):
    dataset = KittiOdometryDataset(data_path=args.data_path, sequence="00")
    slam = NeuralSLAM(args, odometry_weights=weights_file, start_mode="relocalization")
    f = open("test_results/log.txt", 'a')
    f.write("\nRelocalization test\n-------------------\n")

    with torch.no_grad():
        rgb, true_orientation, true_position = dataset[195]
        rgb = TF.resize(rgb, (376, 1232))
        initial_pose, refined_pose, distances = slam(rgb)
        f.write("Distances shape: " + str(distances.shape) + '\n')

        # Histogram
        plt.figure()
        plt.hist(distances.cpu().numpy(), bins=1000)
        plt.xlabel("Distance from sample")
        plt.ylabel("Count of elements")
        plt.savefig("test_results/distance_histogram.png")

        # Distances
        plt.figure()
        plt.plot(distances.cpu().numpy())
        plt.xlabel("Index of keyframe")
        plt.ylabel("Embedding distance from sample")
        plt.savefig("test_results/distances.png")
        
        # Predicted index
        pred_index = torch.argmin(distances)
        distances[pred_index] = distances.max()
        second_pred_index = torch.argmin(distances)
        f.write("Predicted index: " + str(pred_index) + '\n')
        f.write("Predicted secondary index: " + str(second_pred_index) + '\n')

        pred_im = torch.load(slam[int(pred_index.squeeze())].rgb_file_name).permute(1, 2, 0)
        second_pred_im = torch.load(slam[int(second_pred_index.squeeze())].rgb_file_name).permute(1, 2, 0)
        plt.imsave("test_results/predicted_frame.png", pred_im.numpy())
        plt.imsave("test_results/secondary_predicted_frame.png", second_pred_im.numpy())

        initial_orientation, initial_position = to_vectors(initial_pose.to("cpu"))
        refined_orientation, refined_position = to_vectors(refined_pose.to("cpu"))

        f.write("True pose: " + str([true_orientation, true_position]) + '\n\n')
        f.write("Initial estimate: " + str([initial_orientation, initial_position]) + '\n')
        f.write("Refined estimate: " + str([refined_orientation, refined_position]) + '\n\n')

        f.write("Initial difference: " + str([(true_orientation-initial_orientation).abs().sum(), (true_position-initial_position).abs().sum()]) + '\n')
        f.write("Refined difference: " + str([(true_orientation-refined_orientation).abs().sum(), (true_position-refined_position).abs().sum()]) + '\n')
        f.close()

def main():
    args = Arguments.get_arguments()
    sequence_length = 1
    weights_file = "atdn_vslam/checkpoints/11_1_atdnvo_c.pth" # TODO overwrite to actual

    if not os.path.exists("test_results"):
        os.mkdir("test_results")
    f = open("test_results/log.txt", 'w')
    f.write("SLAM test output log \n=====================\n")
    f.close()

    global_pos = test_odometry(args, weights_file)
    test_remapping(args, weights_file, global_pos)
    test_relocalization(args, weights_file)

if __name__ == "__main__":
    main()
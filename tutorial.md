# Getting started
In this tutorial we show a detailed example of how to use ATDN vSLAM.

## Imports, datasets, arguments and SLAM instantiation.
The arguments class is a convenient way to handle general configuration variables.
Using the atdn_vslam datasets is an easy way to show the functionality with the KITTI odometry dataset. If you would like to test ATDN with custom data, the only requirement is to use RGB images (valued between 0 and 255) as float tensors with a shape of (3, 376, 1232).

```python
import os
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


args = Arguments.get_arguments()
weights_file = "atdn_vslam/checkpoints/10_1_atdnvo_c.pth" # TODO overwrite tod actual
dataset = ColorDataset(data_path=args.data_path, sequence="00")

slam = NeuralSLAM(args, odometry_weights=weights_file)
```

## Changing state
After that, SLAM state can be changed from idle to odometry. The actual SLAM state can be accessed through the mode() method.

```python
slam.start_odometry()
print("SLAM mode: ", slam.mode())
```

## Odometry
Next, odometry estimations can be made by calling the SLAM object. In this example, the inference is extended with a simple runtime benchmarking.

```python
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

log("Average odometry time: ", slam_call_time.mean())
log("Odometry time std: ", slam_call_time.std())
log("FPS from time: ", 1/slam_call_time.mean())
```

## Mapping
When the environment is explored, ATDN vSLAM can be changed to mapping by ending the odometry. This will initiate the learning procedure of the general map of registered keyframes.

```python
slam.end_odometry()
```

## Retrieving keyframe data
Indexing the SLAM object gives us a keyframe. Through a keyframe we can acces the keyframe's image path, pose and mapping code.

```python
keyframe_positions = []

for i in range(len(slam)):
    pose = slam[i].pose
    keyframe_positions.append(pose[:3, 3])

keyframe_positions = torch.stack(keyframe_positions, dim=0).to("cpu")
X_key, Y_key, Z_key = keyframe_positions[:, 0], keyframe_positions[:, 1], keyframe_positions[:, 2]

plt.figure()
plt.scatter(X_key, Z_key)
plt.savefig("test_results/keyframe_poses.png")
```

## Relocalization
After mapping, relocalization can be done by calling the SLAM object with the querry image

```python
# You can start from relocalization state if odometry and mapping is done in a previous run
dataset = KittiOdometryDataset(data_path=args.data_path, sequence="00")
slam = NeuralSLAM(args, odometry_weights=weights_file, start_mode="relocalization")

with torch.no_grad():
    rgb, true_orientation, true_position = dataset[195]
    rgb = TF.resize(rgb, (376, 1232))
    initial_pose, refined_pose, distances = slam(rgb)
    log("Distances shape", distances.shape)

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

def to_vectors(mat):
    orientation = matrix2euler(mat[:3, :3])
    position = mat[:3, -1]
    return orientation, position

initial_orientation, initial_position = to_vectors(initial_pose.to("cpu"))
refined_orientation, refined_position = to_vectors(refined_pose.to("cpu"))

log("True pose: ", [true_orientation, true_position])
log("Initial estimate: ", [initial_orientation, initial_position])
log("Refined estimate: ", [refined_orientation, refined_position])

log("Initial difference: ", [(true_orientation-initial_orientation).abs().sum(), 
                             (true_position-initial_position).abs().sum()])

log("Refined difference: ", [(true_orientation-refined_orientation).abs().sum(), 
                             (true_position-refined_position).abs().sum()])
```

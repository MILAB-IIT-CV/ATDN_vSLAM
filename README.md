# DeepLearning-SLAM
This is the implementation of the Deep Neural SLAM algorithm that is an all-through Deep Learning based solution for the vision based Simultaneous Localization and Mapping (SLAM) task.


## Prerequisites
- PyTorch: The main Machine Learning library.
- GMA optical flow library: Used for Deep Learnig based flow estimation (has to be cloned in the SLAM's library to a GMA subfolder)

# Using the implemented SLAM system
Here is an example how to use the SLAM.
### First, imports, dataset, arguments object and SLAM instantiation.
The arguments class is implemented in helpers.py and is a convenient way to handle generally useful functionality and variables.

```python
import torch
from torch.utils.data import DataLoader
from slam_framework.neural_slam import NeuralSLAM
from helpers import Arguments, log
from odometry.vo_datasets import KittiOdometryDataset
import time
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100


args = Arguments()
sequence_length = 1
preprocessed_flow = True
dataset = KittiOdometryDataset(args.data_path, "00", precomputed_flow=preprocessed_flow, sequence_length=sequence_length)

# TODO change to actual data path
data_path = "path/to/data"
slam = NeuralSLAM(data_path, preprocessed_flow=preprocessed_flow)
```

### After that, SLAM mode can be changed from idle to odometry. Actual SLAM mode can be accessed through the mode() methodd

```python
slam.start_odometry()
print("SLAM mode: ", slam.mode())
```

### Next, odometry estimations can be made by calling the SLAM object

```python
global_scale = []
count = 0
slam_call_time = []
for i in range(0, len(dataset), sequence_length):
    if preprocessed_flow:
        img1, img2, flow, _, __ = dataset[i]
        start = time.time()
        current_pose = slam(img1.squeeze(), img2.squeeze(), flow.squeeze())
        end = time.time()
    else:
        img1, img2, _, __ = dataset[i]
        start = time.time()
        current_pose = slam(img1.squeeze(), img2.squeeze())
        end = time.time()
        
    slam_call_time.append(end-start)
    global_scale += current_pose
    
    print('    ', end='\r')
    print(count, end='')
    count = count+sequence_length

global_scale = torch.stack(global_scale, dim=0)
slam_call_time = np.array(slam_call_time)
```

### When the environment is explored, SLAM can be changed to mapping by ending the odometry. This will initialize the learning of the mapping of the registered keyframes

```python
slam.end_odometry()
```

### The keyframe positions are saved under data_path/poses.pht, so it can be plotted with the following

```python
import matplotlib.pyplot as plt
poses = torch.load(data_path+"/poses.pth")
print(poses.shape)
X = poses[:, 3]
Z = poses[:, -1]
plt.scatter(X.numpy(), Z.numpy())
plt.show()
```

### Keyframes can be also obtained by indexing of the SLAM object
### After the mapping relocalization can be done by calling the SLAM object with the image to search

```python
DATA_PATH = None # TODO Change to config file data read
dataset = MappingDataset(DATA_PATH, slam=True)
rgb_mean = torch.load("normalization_cache/rgb_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
rgb_std = torch.load("normalization_cache/rgb_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

dataset = KittiLocalizationDataset(data_path=args.data_path, sequence="00")

rgb, true_orientation, true_position = dataset[195]

padder = InputPadder(rgb.shape)
rgb = padder.pad(rgb)[0]

im_normalized = (rgb.unsqueeze(0)-rgb_mean)/rgb_std

with torch.no_grad():
    initial_pose, refined_pose, distances = slam(im_normalized)

def to_vectors(mat):
    abs_rotation = mat[:3, :3]
    abs_translation = mat[:3, -1]

    orientation = matrix2euler(abs_rotation)
    position = abs_translation
    return orientation, position

initial_orientation, initial_position = to_vectors(initial_pose)
refined_orientation, refined_position = to_vectors(refined_pose)
log("True pose: ", [true_orientation, true_position])
log("Initial estimate: ", [initial_orientation, initial_position])
log("Refined estimate: ", [refined_orientation, refined_position])

log("Initial difference: ", [(true_orientation-initial_orientation).abs().sum(), (true_position-initial_position).abs().sum()])
log("Refined difference: ", [(true_orientation-refined_orientation).abs().sum(), (true_position-refined_position).abs().sum()])
```
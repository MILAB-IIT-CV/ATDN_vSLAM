# How to use the SLAM system
In the following we show a detailed example of how to use the implemented SLAM. A Jupyter notebook file (slam_test.ipynb) is also available to try out the key features of the framework.

### Imports, dataset, arguments object and SLAM instantiation.
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


args = Arguments.get_arguments()

dataset = KittiOdometryDataset(args.data_path, 
                               "00", 
                               precomputed_flow=args.precomputed_flow, 
                               sequence_length=args.sequence_length)

slam = NeuralSLAM(args)
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
    if args.precomputed_flows:
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

### The keyframe positions are saved under data_path/poses.pht thereby can be plotted with the following

```python
import matplotlib.pyplot as plt
DATA_PATH = args.keyframes_path + "/poses.pth"
poses = torch.load(DATA_PATH)
print(poses.shape)
X = poses[:, 3]
Z = poses[:, -1]
plt.scatter(X.numpy(), Z.numpy())
plt.show()
```

### Keyframes can be also obtained by indexing of the SLAM object
### After the mapping relocalization can be done by calling the SLAM object with the image to search

```python
from localization.localization_dataset import MappingDataset, KittiLocalizationDataset
from GMA.core.utils.utils import InputPadder
from helpers import log, matrix2euler
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100

dataset = MappingDataset(args.keyframes_path, slam=True)
rgb_mean = torch.load("normalization_cache/rgb_mean.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
rgb_std = torch.load("normalization_cache/rgb_std.pth").unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

dataset = KittiLocalizationDataset(data_path=args.data_path, sequence="00")

rgb, true_orientation, true_position = dataset[195]

padder = InputPadder(rgb.shape)
rgb = padder.pad(rgb)[0]

im_normalized = (rgb.unsqueeze(0)-rgb_mean)/rgb_std

with torch.no_grad():
    initial_pose, refined_pose, distances = slam(im_normalized)
    #log("Distances shape", distances.shape)

    max_dist = torch.max(distances).cpu().numpy()
    bins = 1000

    [hist, bin_edges] = np.histogram(distances.cpu().numpy(), bins=bins)

    # ---------
    # Histogram
    # ---------

    plt.bar(bin_edges[:-1], hist, width=5)
    plt.xlabel("Distance from sample")
    plt.ylabel("Count of elements")
    plt.show()

    # Predicted index
    distances_mean = distances.mean()
    pred_index = torch.argmin(distances)
    second_pred_index = torch.argmin(distances)

    plt.plot(distances.cpu().numpy())
    plt.xlabel("Index of keyframe")
    plt.ylabel("Embedding distance from sample")
    plt.show()

def prepare_im(im):
    return im.detach().byte().squeeze().permute(1, 2, 0).numpy()


pred_im, _, _ = dataset[int(pred_index.squeeze())]
second_pred_im, _, _ = dataset[int(second_pred_index.squeeze())]

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

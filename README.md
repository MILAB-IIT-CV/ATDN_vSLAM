# ATDN vSLAM
This is the implementation of the ATDN vSLAM algorithm ([paper](https://pp.bme.hu/eecs/article/view/20437)) that is an all-through Deep Learning based solution for the vision based Simultaneous Localization and Mapping (SLAM) task.

If you are using this work please cite our paper:
```bibtex
@article{Szanto_Bogar_Vajta_2022, 
    title={ATDN vSLAM: An All-Through Deep Learning-Based Solution for Visual Simultaneous Localization and Mapping},
    volume={66},
    url={https://pp.bme.hu/eecs/article/view/20437},
    number={3},
    journal={Periodica Polytechnica Electrical Engineering and Computer Science},
    author={Szántó, Mátyás and Bogár, György Richárd and Vajta, László},
    year={2022},
    pages={236–247}
}
```

## Prerequisites
- PyTorch: The main Machine Learning library.
- [GMA](https://github.com/zacjiang/GMA) optical flow library: Used for Deep Learnig based flow estimation (has to be cloned in the SLAM's library to a GMA subfolder. After cloning, imports have to be updated.)
- For GMA, [einops](https://github.com/arogozhnikov/einops) is also required.

# Using the implemented SLAM system
In the following there is a detailed example of how to use the SLAM. A Jupyter notebook file (slam_test.ipynb) is also available to try out the key features of the framework.

 Before using the SLAM, a .yaml config file is required for the Arguments object creation. Here is an example how config.yaml should look like:
 ```yaml
!!python/object:arguments.Arguments
alpha: 0.3
batch_size: 8
data_path: /path/to/dataset
device: cuda:0
epochs: 2
epsilon: 1.0e-08
keyframes_path: /path/to/keyframes
weight_file: odometry/clvo_general_
log_file: loss_log/generalization_
lr: 0.001
stage: 6
sequence_length: 8
train_sequences:
- '00'
- '01'
- '02'
- '03'
- '04'
- '06'
- '08'
- '09'
- '10'
wd: 0.0001
weight_decay: false
precomputed_flow: true
w : 3

 ```

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

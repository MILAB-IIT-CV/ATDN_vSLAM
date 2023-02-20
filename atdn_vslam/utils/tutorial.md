# How to use the SLAM system
In this tutorial we show a detailed example of how to use ATDN vSLAM.

### Imports, dataset, arguments object and SLAM instantiation.
The arguments class is a convenient way to handle general configuration variables.

```python
import glob
import time

from tqdm.notebook import trange
import torch
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
import matplotlib.pyplot as plt

from slam_framework.neural_slam import NeuralSLAM
from utils.helpers import log
from utils.arguments import Arguments
from localization.datasets import ColorDataset


args = Arguments.get_arguments()
sequence_length = 1

dataset = ColorDataset(data_path=args.data_path, sequence="00")

weights_file = "checkpoints/10_1atdnvo_c.pth" # TODO overwrite tod actual
slam = NeuralSLAM(args, odometry_weights=weights_file)
```

### After that, SLAM state can be changed from idle to odometry. The actual SLAM state can be accessed through the mode() method.

```python
slam.start_odometry()
print("SLAM mode: ", slam.mode())
```

### Next, odometry estimations can be made by calling the SLAM object. In this example, the inference is extended with a simple runtime benchmarking.

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
fps_manual = 1/(slam_call_time.mean())
log("FPS from time: ", 1/slam_call_time.mean())
```

### When the environment is explored, ATDN vSLAM can be changed to mapping by ending the odometry. This will initiate the learning procedure of the general map of registered keyframes.

```python
slam.end_odometry()
```

### Indexing the SLAM object gives us a keyframe. Through a keyframe we can acces the keyframe's image path, pose and mapping code.

```python
keyframe_positions = []

for i in range(len(slam)):
    pose = slam[i].pose
    keyframe_positions.append(pose[:3, 3])

keyframe_positions = torch.stack(keyframe_positions, dim=0).to("cpu")
X_key, Y_key, Z_key = keyframe_positions[:, 0], keyframe_positions[:, 1], keyframe_positions[:, 2]

plt.scatter(X_key, Z_key)
plt.show()
```

### After mapping, relocalization can be done by calling the SLAM object with the querry image

```python
import time

import torch
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100

from utils.helpers import log
from utils.transforms import matrix2euler
from utils.arguments import Arguments
from odometry.datasets import KittiOdometryDataset
from slam_framework.neural_slam import NeuralSLAM


args = Arguments.get_arguments()
dataset = KittiOdometryDataset(data_path=args.data_path, sequence="00")

weights_file = "checkpoints/10_1atdnvo_c.pth" # TODO overwrite to actual
slam = NeuralSLAM(args, odometry_weights=weights_file, start_mode="relocalization")

with torch.no_grad():
    rgb, true_orientation, true_position = dataset[195]
    rgb = TF.resize(rgb, (376, 1232))
    initial_pose, refined_pose, distances = slam(rgb)
    log("Distances shape", distances.shape)

    # Histogram
    plt.hist(distances.cpu().numpy(), bins=1000)
    plt.xlabel("Distance from sample")
    plt.ylabel("Count of elements")
    plt.show()

    plt.plot(distances.cpu().numpy())
    plt.xlabel("Index of keyframe")
    plt.ylabel("Embedding distance from sample")
    plt.show()
    
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
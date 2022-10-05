# ATDN vSLAM
This is the implementation of the ATDN vSLAM algorithm ([paper](https://pp.bme.hu/eecs/article/view/20437)) that is an all-through Deep Learning based solution for the vision based Simultaneous Localization and Mapping (SLAM) task.

In case you are using this work please cite our paper:
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

### Config

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
## Usage

For a detailed introduction to the system usage, please check our [tutorial](utils/tutorial.md).
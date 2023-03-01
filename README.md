# ATDN vSLAM
This is the implementation of ATDN vSLAM ([paper](https://pp.bme.hu/eecs/article/view/20437)) that is an all-through Deep Learning based solution for the vision based Simultaneous Localization and Mapping (SLAM) task.

![System architecture](ATDN_vSLAM.png)

In case you are using this work, please cite our paper:
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

## Setup
1. Venv & updates
```bash
# optional if you want to work in a virtual environment
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools
```
2. Clone the repo and go to its library
```bash
git clone https://github.com/MILAB-IIT-CV/ATDN_vSLAM.git
cd ATDN_vSLAM
```
3. Finally, you can install ATDN vSLAM and its dependencies locally via
```bash
python3 -m pip install GMA-1.0.0-py3-none-any.whl
python3 -m pip install -e .
```
4. Add yaml config as described in the following section

## Config

 Before using the SLAM, create a config.yaml file under atdn_vslam/utils. Here is an example what it should contain:
 
> **Note** : The fields "_data_path_" and "_keyframes_path_" have to be changed! 
> 
> Currently, only the KITTI odometry dataset is supported so "_data_path_" has to be set to the path where its _dataset_ folder is placed. You can download the KITTI dataset from the official [KITTI website](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). 
> 
> "_keyframes_path_" is the path where you want ATDN_vSLAM to save its outputs.

 ```yaml
!!python/object:utils.arguments.Arguments
alpha: 1
batch_size: 24
data_path: /path/to/dataset
device: cuda:0
epochs: 1
epsilon: 1.0e-08
keyframes_path: /path/to/SLAM/output
weight_file: atdn_vslam/checkpoints/clvo_generalization4_
log_file: loss_log/generalization4_
lr: 0.01
stage: 1
sequence_length: 6
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
wd: 0.001
augment_flow: false
w : 2
 ```

## Usage

For a detailed introduction to the system usage, please check our [tutorial](tutorial.md).

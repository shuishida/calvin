# Towards real-world navigation with deep differentiable planners
Shu Ishida, João F. Henriques (Visual Geometry Group, University of Oxford)

![CALVIN has been tested in a grid maze environment, MiniWorld and with the Active Vision Dataset](https://user-images.githubusercontent.com/16188477/171409595-75b53424-5038-48bf-9c70-3b4144f1e09e.png)

## Overview
This is the official code for [the paper](https://arxiv.org/abs/2108.05713): S. Ishida, J. F. Henriques "Towards real-world navigation with deep differentiable planners", CVPR 2022.
This code base allows us to easily switch between variations of VIN-like deep differentiable planners (VIN, GPPN and CALVIN) and different training environments (grid world, MiniWorld and AVD).

### Blog post
Please check out the [official blog post on CALVIN](https://www.robots.ox.ac.uk/~vgg/blog/calvin-a-neural-network-that-can-learn-to-plan-and-navigate-unknown-environments.html).

### Citing
If you find this code base or models useful in your research, please consider citing the following paper:

```
@inproceedings{ishida2022calvin,
    title = {Towards real-world navigation with deep differentiable planners},
    author = {Ishida, Shu and Henriques, João F.},
    year = {2022},
    month = jun,
    booktitle = {2022 {IEEE} {Conference} on {Computer} {Vision} and {Pattern} {Recognition} ({CVPR})},
    publisher = {IEEE},
}
```

### Structure of repository
In the `core` directory, you will find `domains` which have the domain-specific environment setups, and `models` which contains deep planners and other neural network models.

The scripts that you tend to use are `core/domains/<domain>/dataset.py` for generating trajectory datasets, 
`core/domains/<domain>/trainer.py` for training the agent, using the trajectory datasets,
`core/scripts/eval.py` to perform rollouts of pre-trained models, and
`core/scripts/visualise.py` to visualise saved rolled out experiences.  

In order to generate these datasets and train the agent, `dataset.py` assumes an implementation of `env.py` for each domain, which has methods to reset the environment, updating the environment upon the agent taking an action, and retrieving observations.
Both `dataset.py` and `trainer.py` instantiate an agent defined in `core/agent.py`, which interacts with the environment.

When the agent takes actions in environments, it collects episodes of experiences. An experience manager is defined in `core/experiences.py`.

Domain-specific pre-processing and post-processing of data (e.g. observations) are handled by `core/domains/<domain>/handler.py`.

While `core/domains/<domain>/trainer.py` extends `core/agent_trainer.py`, which standardises both RL-like training and Imitation Learning-like training, and handles the instantiation of both the agent and the model, 
if you want to experiment with different training setups, there are some example scripts provided in `core/domains/<domain>/examples/`.

In the `scripts` directory, you can find commands to generate training trajectories and train the agent models.
Upon running the dataset generation script, the trajectories will be stored under the `data` directory.

## Setup 
Clone repository
```bash
git clone https://github.com/shuishida/calvin.git
cd calvin
```

### Create Conda environment
```bash
conda create -n calvin python=3.7
conda activate calvin
```
Some of the scripts (e.g. pptk) require the path to libstdc++ and libz that comes with conda.
Run this before launching the script if you encounter errors such as missing libstdc++ files.
```bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
```

### Install dependencies
##### PIP installable Python packages
```bash
pip install numpy matplotlib overboard einops torch-scatter numba tensorboard
```

##### [PyTorch](https://pytorch.org/)
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

##### [Gym MiniWorld](https://github.com/maximecb/gym-miniworld) (Skip if not using MiniWorld environment)
```bash
mkdir third_party
cd third_party
git clone https://github.com/maximecb/gym-miniworld.git
cd gym-miniworld
pip install -e .
cd ../../
```

##### [Active Vision Dataset](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/index.html) (Skip if not using AVD environment)
In order to perform experiments for AVD, you need to download the dataset from the [AVD website](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/index.html), and place it under `./data/avd/src`.
You also need to place `./data/avd/src/train.txt` and `./data/avd/src/val.txt`, which are text files defining which scenes you want to use as train and val scenes, where scene names are separated by new lines.
Please make sure that each scene subdirectory contains a `cameras.txt` file, which provides the camera parameters. If these are not available, please contact the authors of the AVD paper.

### Domains
Currently, we have
- 2D grid world
    - Maze Map (randomly generated maze in 2D)
    - Obstacle Map (randomly generated obstacles occupy the 2D grid world)
- MiniWorld
- Active Vision Dataset

### Generate trajectory datasets
To generate maps with randomised start and target positions and corresponding optimal trajectories, 
run the `dataset.py` script for each domain in the `core/domains/<domain>` directory.

For example, for the 2D grid world domain, we can generate the maze or obst dataset by:
```bash
python core/domains/gridworld/dataset.py --map <maze | obst> -sz 15 15 -vr <view range> -trajlen 15 -mxs <max rollout steps> -n 4000 [--ego] [--clear]
```

Here, view range defines the radius the agent is able to view. 
If this argument is not passed, the agent acquires the entire view of the map.
Passing `--ego` will make the state space 2D position + orientation, rather than just position.
For more information about the parameters that can be passed, see help by passing `-h` as an argument.
By default, the generated data will be saved at `data/<domain>/`.

##### Examples
```bash
python core/domains/gridworld/dataset.py --map maze -sz 15 15 -vr 2 -trajlen 15 -mxs 500 -n 4000 --clear
python core/domains/gridworld/dataset.py --map maze -sz 15 15 -vr 2 -trajlen 15 -mxs 500 -n 4000 --ego --clear
```

Examples of how to run the scripts can be found in the comments at the bottom of each script.

### Training model

The models can be trained by:
```bash
python core/domains/gridworld/trainer.py --data <path to data directory> --model <name of model> --k <k> --discount <discount rate> --n_workers 4
```
By default, the trained model will be saved in the same directory as the data directory.

More parameters might be required depending on which model you train. 

```bash
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_vr_2_4000_15_500 --model CALVINConv2d --k 60 --discount 0.25
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_vr_2_4000_15_500_ego --model CALVINConv3d --k 20 --discount 0.1
```

Examples of how to run the scripts can be found in the comments at the bottom of each script.

### Performing rollout evaluation
```bash
python core/scripts/eval.py <path/to/checkpoint>/checkpoint.pt -nev <number of evaluations>
```
This will load the pre-trained parameters saved as `checpoint.pt`, instantiating an agent and model using the config saved as `checkpoint.pt.json`, and perform a number of rollouts, and saving the experiences in `<path/to/checkpoint_eval/>`.

### Inspecting the model
```bash
python core/scripts/visualise.py <path/to/checkpoint_eval/> <plotter_name>
```

You can write your own plot function and 

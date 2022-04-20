#!/bin/bash

## Fully-known
### CALVIN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_4000_15_200 --model CALVINConv2d --k 100 --l_i 2 --sparse --n_workers 4 $1
### VIN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_4000_15_200 --model VIN --k 40 --l_i 2 --l_q 40 -pn --sparse --n_workers 4 $1
### GPPN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_4000_15_200 --model GPPN --k 20 --k_sz 9 --l_i 2 --sparse --n_workers 4 $1

# Partial view positional
### CALVIN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_vr_2_4000_15_500 --model CALVINConv2d --k 60 --sparse --n_workers 4 $1
### VIN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_vr_2_4000_15_500 --model VIN --k 20 --l_q 40 -pn --sparse --n_workers 4 $1
### GPPN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_vr_2_4000_15_500 --model GPPN --k 20 --k_sz 5 --sparse --n_workers 4 $1

# Partial view embodied
### CALVIN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_vr_2_4000_15_500_ego --model CALVINConv3d --k 20 --sparse --n_workers 4 $1
### VIN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_vr_2_4000_15_500_ego --model VINPose --k 20 --l_q 40 -pn --sparse --n_workers 4 $1
### GPPN
python core/domains/gridworld/trainer.py --data data/gridworld/MazeMap_15x15_vr_2_4000_15_500_ego --model GPPNPose --k 20 --k_sz 5 --sparse --n_workers 4 $1

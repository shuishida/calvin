#!/bin/bash

## Small maze 3x3
### CALVIN
python core/domains/miniworld/trainer.py --data data/miniworld/Maze_3__0_0_10_10__30_30__8_1000_0_300_8/ --model CALVINPoseNav --k_sz 5 --k 40 -ms 10 -sm 1 --pcn_h 80 --l_h 80 --pcn_f 40 --lr 0.001 -bs 32 --dropout 0.2 --discount 0.1 --clip 0.1 --n_workers 4 $1
### VIN
python core/domains/miniworld/trainer.py --data data/miniworld/Maze_3__0_0_10_10__30_30__8_1000_0_300_8/ --model VINPoseNav --k_sz 5 --k 20 -ms 10 -sm 1 --pcn_h 80 --l_h 80 --pcn_f 40 --lr 0.001 -bs 32 --dropout 0.2 --clip 0.1 --l_q 40 -pn --discount 0.1 --n_workers 4 $1
### GPPN
python core/domains/miniworld/trainer.py --data data/miniworld/Maze_3__0_0_10_10__30_30__8_1000_0_300_8/ --model GPPNPoseNav --k_sz 5 --k 20 -ms 10 -sm 1 --pcn_h 80 --l_h 80 --pcn_f 40 --lr 0.001 -bs 32 --dropout 0.2 --clip 0.1 --discount 0.1 --n_workers 4 $1

# Large maze 8x8
### CALVIN
python core/domains/miniworld/trainer.py --data data/miniworld/Maze_8__-1_-1_27_27__80_80__8_1000_0_1000_8/ --model CALVINPoseNav --k_sz 3 --k 40 -ms 10 -sm 1 --pcn_h 80 --l_h 80 --pcn_f 40 -bs 32 -frames 40 --dropout 0.2 --clip 0.1 --lr 0.001 --discount 0.1 --n_workers 4 $1
### VIN
python core/domains/miniworld/trainer.py --data data/miniworld/Maze_8__-1_-1_27_27__80_80__8_1000_0_1000_8/ --model VINPoseNav --k_sz 3 --k 20 -ms 10 -sm 1 --pcn_h 80 --l_h 80 --pcn_f 40 -bs 32 -frames 40 --dropout 0.2 --clip 0.1 --lr 0.001 --l_q 40 -pn --discount 0.1 --n_workers 4 $1
### GPPN
python core/domains/miniworld/trainer.py --data data/miniworld/Maze_8__-1_-1_27_27__80_80__8_1000_0_1000_8/ --model GPPNPoseNav --k_sz 5 --k 20 -ms 10 -sm 1 --pcn_h 80 --l_h 80 --pcn_f 40 -bs 32 -frames 40 --dropout 0.2 --clip 0.1 --lr 0.001 --discount 0.1 --n_workers 4 $1


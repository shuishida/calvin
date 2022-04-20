#!/bin/bash

### CALVIN
python core/domains/avd/trainer.py --data data/avd/pose_nav/1000_-15_-15_15_15__40_40__12_68_120_coca_cola_glass_bottle_10_100_8/ --model CALVINPoseNav --k_sz 5 --k 10 -ms 10 -sm 1 --pcn_h 40 --l_h 40 --pcn_f 20 --discount 0.1 --lr 0.001 -bs 8 --dropout 0.2 --clip 0.1 -env 25 -nevt 100 -trajlen 0 -bn --epochs 20
### VIN
python core/domains/avd/trainer.py --data data/avd/pose_nav/1000_-15_-15_15_15__40_40__12_68_120_coca_cola_glass_bottle_10_100_8/ --model VINPoseNav --k_sz 5 --k 10 -ms 10 -sm 1 --pcn_h 40 --l_h 40 --pcn_f 20 --discount 0.1 --lr 0.001 -bs 8 --dropout 0.2 --clip 0.1 -env 25 -nevt 100 -trajlen 0 -bn --l_q 40 -pn --epochs 20
### GPPN
python core/domains/avd/trainer.py --data data/avd/pose_nav/1000_-15_-15_15_15__40_40__12_68_120_coca_cola_glass_bottle_10_100_8/ --model VINPoseNav --k_sz 5 --k 10 -ms 10 -sm 1 --pcn_h 40 --l_h 40 --pcn_f 20 --discount 0.1 --lr 0.001 -bs 8 --dropout 0.2 --clip 0.1 -env 25 -nevt 100 -trajlen 0 -bn --epochs 20

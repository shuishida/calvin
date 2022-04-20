#!/bin/bash

python core/domains/gridworld/dataset.py --map maze -sz 15 15 -trajlen 15 -mxs 200 -n 4000 --clear
python core/domains/gridworld/dataset.py --map maze -sz 15 15 -vr 2 -trajlen 15 -mxs 500 -n 4000 --clear
python core/domains/gridworld/dataset.py --map maze -sz 15 15 -vr 2 -trajlen 15 -mxs 500 -n 4000 --ego --clear


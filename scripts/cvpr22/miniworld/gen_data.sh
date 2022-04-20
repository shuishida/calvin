#!/bin/bash

python core/domains/miniworld/dataset.py -sz 3 --map_res 30 30 --costmap_margin 5 -mxs 300 -ori 8 --clear
python core/domains/miniworld/dataset.py -sz 8 --map_res 80 80 --map_bbox -1 -1 27 27 --costmap_margin 5 -mxs 1000 -ori 8 --clear

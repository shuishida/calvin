#!/bin/bash

python core/domains/avd/dataset.py -mxs 100 -trajlen 10 -ori 12 --resize 68 120 --clear
python core/domains/avd/dataset.py -mxs 100 -trajlen 10 -ori 12 --resize 68 120 --target coca_cola_glass_bottle --clear


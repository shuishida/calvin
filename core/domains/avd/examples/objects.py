import os
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("data")
args = parser.parse_args()

item_scene = defaultdict(list)

for mode in ['train', 'val']:
    subdir = os.path.join(args.data, mode)
    scene_names = [d for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))]
    for scene_name in sorted(scene_names):
        scene_path = os.path.join(subdir, scene_name)
        with open(os.path.join(scene_path, "present_instance_names.txt"), 'r') as f:
            for item in f.read().split('\n'):
                item_scene[item].append(scene_name)
items = sorted([(len(v), k, v) for k, v in item_scene.items()])[::-1][:5]
print(items)

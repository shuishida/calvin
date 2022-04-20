import os
from collections import defaultdict
from typing import Dict

import numpy as np

from core.domains.avd.dataset.data_classes import Scene, ObjectClass
from core.domains.avd.dataset.scene_loader import avd_scene_loader
from core.utils.tensor_utils import random_choice


def get_states_in_range(target_class: ObjectClass, target_size_ratio=0.6):
    if len(target_class) == 0:
        return []
    max_obj_size = np.array([obj.size for obj in target_class]).max()
    target_objects = [obj for obj in target_class if obj.size >= max_obj_size * target_size_ratio]
    return target_objects


class AVDSceneManager(dict):
    def __init__(self, data_dir: str, scene_resize: tuple, target: str = None, in_ram=False,
                 target_size_ratio=None, target_dist_thresh=None, avd_workers=None, **kwargs):
        super(AVDSceneManager, self).__init__()
        self.data_dir = data_dir
        self.scene_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.target_objects_dict = defaultdict(dict)
        self.target_name = target

        for scene_name in self.scene_names:
            scene = avd_scene_loader(data_dir, scene_name, scene_resize, in_ram, avd_workers)
            self[scene_name] = scene
        self.target_size_ratio = target_size_ratio
        self.target_dist_thresh = target_dist_thresh
        self.reset_targets(target)

        self.train = self.get_split("train")
        self.val = self.get_split("val")
        self.split = {'train': self.train, 'val': self.val}

    def get_split(self, split: str) -> Dict[str, Scene]:
        with open(os.path.join(self.data_dir, f"{split}.txt"), "r") as f:
            scene_names = f.read()
        scene_names = scene_names.strip().split("\n")
        return {k: self[k] for k in scene_names}

    def reset_targets(self, target_name=None):
        self.target_objects_dict = defaultdict(dict)
        for scene_name in self.scene_names:
            print(f"Finding targets for scene {scene_name}")
            scene = self[scene_name]
            target_classes = [
                scene.object_classes.get_by_name(target_name)] if target_name else scene.object_classes.values()
            for target_class in target_classes:
                target_objects = get_states_in_range(target_class, target_size_ratio=self.target_size_ratio)
                if len(target_objects):
                    self.target_objects_dict[scene_name][target_class.name] = target_objects

    def select_targets(self, scene_name):
        objects = list(self.target_objects_dict[scene_name].keys())
        target_name = random_choice(objects)
        target_objects = self.target_objects_dict[scene_name][target_name]
        return target_name, target_objects

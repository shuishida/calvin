import os

import numpy as np

from core.domains.avd.dataset.scene_manager import AVDSceneManager
from core.domains.avd.navigation.pos_nav.actions import AVDPosActionSet
from core.domains.avd.navigation.pose_nav.actions import AVDPoseActionSet
from core.mdp.meta import MDPMeta
from core.mdp.planner import Planner
from core.utils.tensor_utils import random_choice
from core.domains.avd.dataset.data_classes import ImageNode


class AVDMDPMetaBase(MDPMeta):
    def __init__(self, scenes: AVDSceneManager, map_bbox: tuple, map_res: tuple, ori_res = None, **kwargs):
        self.map_bbox = map_bbox
        self.map_res = map_res
        self.ori_res = ori_res
        self.scenes = scenes
        super(AVDMDPMetaBase, self).__init__(AVDPoseActionSet() if ori_res else AVDPosActionSet(),
                                             (ori_res, *map_res) if ori_res else map_res)

    def get_states_from_scene(self, scene_name):
        raise NotImplementedError

    def get_states_from_objects(self, objects):
        raise NotImplementedError


class AVDPlannerBase(Planner):
    def __init__(self, meta: AVDMDPMetaBase, scene_name):
        self.scene = meta.scenes[scene_name]
        states = meta.get_states_from_scene(scene_name)
        super(AVDPlannerBase, self).__init__(meta, states)

        self.target_name, self.target_objects = self.meta.scenes.select_targets(scene_name)
        target_states = meta.get_states_from_objects(self.target_objects)
        self.target_states = list(set(target_states))
        if len(self.target_states) == 0:
            raise Exception("No valid target classes found")

    def terminal_condition(self, state, target):
        return state in self.target_states

    @property
    def grid(self):
        grid = np.zeros(self.meta.map_res)
        for state in self.states:
            grid_index = self.meta.state_to_grid_index(state)
            grid[grid_index] = 1
        return grid

    def targets_grid(self):
        grid = np.zeros(self.meta.map_res, dtype=np.uint8)
        if self.target_states is None:
            raise Exception("Targets grid can only be generated when the target class is defined.")
        for target in self.target_states:
            grid_index = self.meta.state_to_grid_index(target)
            grid[grid_index] = 1
        return grid

    def sample_trajectories(self, min_traj_len, n_trajs=1, target: ImageNode=None):
        """
        :param min_traj_len: minimum threshold for trajectory length
        :param n_trajs: number of trajectories to be sampled
        :param target: target state
        :return: list of trajectories, where each trajectory is a list of tuples (state, action) along trajectory
        """
        if target is None and self.target_states is not None:
            target = random_choice(self.target_states)
        return super().sample_trajectories(min_traj_len=min_traj_len, n_trajs=n_trajs, target=target)

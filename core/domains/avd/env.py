from typing import Optional, Tuple, Any, List, Dict, Union

import torch
from einops import rearrange

from core.domains.avd.dataset.data_classes import Scene
from core.domains.avd.navigation.pos_nav.pos_planner import AVDPosMDPMeta, AVDPosPlanner
from core.domains.avd.navigation.pose_nav.pose_planner import AVDPoseMDPMeta, AVDPosePlanner
from core.env import Env
from core.utils.env_utils import NavStatus
from core.utils.image_utils import square_resize
from core.utils.tensor_utils import random_choice


class AVDEnv(Env):
    def __init__(self, meta: Union[AVDPosMDPMeta, AVDPoseMDPMeta], *, split=None,
                 min_traj_len=0, max_steps=None, sample_free=None, done_explicit=None, **kwargs):
        self.min_traj_len = min_traj_len
        self.sample_free = sample_free

        self.split = split
        self.scenes: Dict[str, Scene] = meta.scenes.split[split]
        self.scene_name_gen = self.gen_scene_name()

        self.state = self.target = self.opt_traj = None
        self.planner: Union[AVDPosPlanner, AVDPosePlanner] = None

        self.done_explicit = done_explicit

        self.target_resize = square_resize((64, 64))
        super(AVDEnv, self).__init__(meta, max_steps)

    def gen_scene_name(self):
        while True:
            for scene_name in self.scenes:
                yield scene_name

    def _reset(self) -> Tuple[dict, Any, Optional[List[Any]]]:
        """
        :return: tuple of episode_info (dict) and initial observation
        """
        self.opt_traj, self.planner = self._reset_traj()
        traj_states, opt_actions, _ = list(zip(*self.opt_traj))
        self.state = traj_states[0]
        target = traj_states[-1]
        target_object = random_choice(self.planner.target_objects)
        target_rgb = rearrange(torch.from_numpy(target_object.rgb()), "h w f -> f h w")
        return {
                   'scene_name': self.planner.scene.name,
                   'target': torch.tensor(self.meta.state_to_grid_index(target)).long(),
                   'targets_grid': torch.from_numpy(self.planner.targets_grid()).bool(),
                   'target_name': self.planner.target_name,
                   'target_rgb': self.target_resize(target_rgb),
                   'target_emb': target_object.embedding(),
                   # 'target_image': target_object.rgb(),
                   'occupancy': self.planner.grid
               }, self.obsv(), opt_actions

    def obsv(self):
        pose = self.meta.state_to_index(self.state)
        return {
            'state_info': self.state.image_name if self.meta.ori_res else repr(self.state),
            'poses': torch.tensor(pose).float()
        }

    def _step(self, action) -> Tuple[Any, float, bool, Any]:
        """
        :param action:
        :return: (obsv, reward, done, info)
        """
        status, reward, done = NavStatus.in_progress, 0, False
        if self.state in self.planner.target_states:
            if action == self.meta.actions.done or not self.done_explicit:
                status, reward, done = NavStatus.success, 1, True
                reward = 1
        if action != self.meta.actions.done:
            trans = self.planner.transition(self.state, action)
            if trans is not None:
                next_state, _, _ = trans
                self.state = next_state
        return self.obsv(), reward, done, {'status': status}

    def _init_planner(self, scene_name) -> Union[AVDPosPlanner, AVDPosePlanner]:
        return AVDPosePlanner(self.meta, scene_name) if self.meta.ori_res else AVDPosPlanner(self.meta, scene_name)

    def _reset_traj(self):
        scene_name = next(self.scene_name_gen)
        trajs = []
        planner = None
        while not trajs:
            planner = self._init_planner(scene_name)
            trajs = planner.sample_trajectories(self.min_traj_len, n_trajs=1)
        return trajs[0], planner

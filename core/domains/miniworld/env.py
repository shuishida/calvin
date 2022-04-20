from typing import Optional, Tuple, Any, List

import numpy as np
import torch
from einops import rearrange

from core.env import Env
from core.utils.env_utils import NavStatus
from core.mdp.meta import MDPMeta
from core.domains.miniworld.planner import MiniWorldPlanner, MiniWorldTrajState, update_miniworld_env


class MiniWorldEnv(Env):
    def __init__(self, meta: MDPMeta, env, *, min_traj_len=0, costmap_margin=None,
                 max_steps=None, sample_free=None, full_view=None):
        self.env = env
        self.min_traj_len = min_traj_len
        self.costmap_margin = costmap_margin
        self.sample_free = sample_free
        self.full_view = full_view

        self.state = self.target = self.opt_traj = None
        self.planner: MiniWorldPlanner = None
        super(MiniWorldEnv, self).__init__(meta, max_steps)

    def _reset(self) -> Tuple[dict, Any, Optional[List[Any]]]:
        """
        :return: tuple of episode_info (dict) and initial observation
        """
        self.opt_traj, self.planner = self._reset_traj()
        traj_states, opt_actions, _ = list(zip(*self.opt_traj))
        self.state = traj_states[0]
        target = traj_states[-1]
        return {
                   'target': torch.tensor(self.meta.state_to_grid_index(target)).long(),
                   'occupancy': self.planner.grid
               }, self.obsv(), opt_actions

    def obsv(self):
        env = self.state.env()
        if self.full_view:
            rgb, _, coords, free_xyz = self.state.get_full_view(self.sample_free)
            rgb = np.concatenate(rgb)
            coords = np.concatenate(coords)
            # free_xyz = np.concatenate(free_xyz)
        else:
            rgb = self.state.rgb
            coords = self.state.world_coords
            # free_xyz = self.state.sample_free_space(n_samples_per_pixel=self.sample_free)
        rgb = torch.from_numpy(rgb).int()
        surf_xyz = torch.from_numpy(coords).float()
        # free_xyz = torch.from_numpy(free_xyz).float()
        pose = self.meta.state_to_index(self.state)
        return {
            'top_view': torch.from_numpy(env.render_top_view()),
            'rgb': rearrange(rgb, "h w f -> f h w"),
            'surf_xyz': rearrange(surf_xyz, "h w f -> f h w"),
            # 'free_xyz': free_xyz,
            'poses': torch.tensor(pose).float()
        }

    def _step(self, action) -> Tuple[Any, float, bool, Any]:
        """
        :param action:
        :return: (obsv, reward, done, info)
        """
        env = self.state.env()
        if self.meta.ori_res:
            _, reward, done, _ = self.env.step(action)
        else:
            _, reward, done, _ = update_miniworld_env(self.meta, env, action)
        self.state = MiniWorldTrajState(env, self.state, is_terminal=done and reward > 0)
        if done:
            if reward > 0:
                status = NavStatus.success
            else:
                status = NavStatus.max_step_exceeded
        else:
            status = NavStatus.in_progress
        return self.obsv(), reward, done, {'status': status}

    def _init_planner(self) -> MiniWorldPlanner:
        self.env.reset()
        return MiniWorldPlanner(self.meta, self.env, costmap_margin=self.costmap_margin)

    def _reset_traj(self):
        traj = []
        planner = None
        while len(traj) <= self.min_traj_len:
            planner = self._init_planner()
            traj = planner.get_trajectory()

        self.env = planner.source.env()
        return traj, planner

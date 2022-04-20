from typing import Tuple, Any, Optional, List

from core.domains.gridworld.agent_view.agent_map import AgentMap
from core.domains.gridworld.planner.gridmap_planner import GridMapPlanner
from core.env import Env
from core.utils.env_utils import check_repeat, NavStatus
from core.mdp.grid_planner import GridPlanner
from core.mdp.meta import MDPMeta
from core.domains.gridworld.map.gridmap import GridMap
from core.domains.gridworld.planner.ego_grid_planner import EgoGridPlanner

import torch


class GridEnv(Env):
    def __init__(self, meta: MDPMeta, gridmap: GridMap, *,
                 min_traj_len=0, view_range=None, ego=False, allow_backward=True, target_known=False, max_steps=None):
        super(GridEnv, self).__init__(meta, max_steps)
        self.gridmap = gridmap
        self.min_traj_len = min_traj_len
        self.ego = ego
        self.view_range = view_range
        self.allow_backward = allow_backward
        self.target_known = target_known
        self.state = self.target = self.opt_traj = None
        self.pred_traj = []
        self.planner: GridPlanner = None
        self.feature_map = None

    def _reset(self) -> Tuple[dict, Any, Optional[List[Any]]]:
        """
        :return: tuple of episode_info (dict) and initial observation
        """
        self.feature_map = None
        self.opt_traj, self.planner = self._reset_traj()
        traj_states, opt_actions, _ = list(zip(*self.opt_traj))
        self.target = target = traj_states[-1]
        self.state = traj_states[0]
        self.pred_traj.append(self.state)
        values, best_action_maps, counts, motions = self.planner.get_values_and_best_actions(target)
        valid_actions = self.planner.get_valid_actions(target)
        return {
            'values': torch.from_numpy(values).float(),
            'best_action_maps': torch.from_numpy(best_action_maps).bool(),
            'valid_actions': torch.from_numpy(valid_actions).bool(),
            'motions': torch.from_numpy(motions).int(),
            'counts': torch.from_numpy(counts).int(),
            'target': torch.tensor(self.meta.state_to_grid_index(target)).long()
        }, self.obsv(), opt_actions

    def obsv(self):
        grid_index = self.meta.state_to_grid_index(self.state)
        pose = self.meta.state_to_index(self.state)
        agent_map = AgentMap(self.planner.grid, self.meta.state_to_grid_index(self.target),
                             self.view_range, target_known=self.target_known)
        agent_map.update(grid_index)
        obsv_map, _ = agent_map.get_obsv()
        # local_obsv = extract_view(obsv_map, *grid_index, self.view_range)
        obsv_map = torch.from_numpy(obsv_map).float()
        self.feature_map = obsv_map if self.feature_map is None else torch.maximum(obsv_map, self.feature_map)
        return {
            'feature_map': self.feature_map,
            'obsv_maps': obsv_map,
            # 'obsv_locals': torch.from_numpy(local_obsv).float(),
            'poses': torch.tensor(pose).float()
        }

    def _step(self, action) -> Tuple[Any, float, bool, Any]:
        """
        :param action:
        :return: (obsv, reward, done, info)
        """
        if action == self.meta.actions.done:
            completed = self.meta.state_to_grid_index(self.state) == self.meta.state_to_grid_index(self.target)
            status = NavStatus.success if completed else NavStatus.false_complete
        else:
            next_state_info = self.planner.transition(self.state, action)
            if next_state_info:
                next_state, action, cost = next_state_info
                self.state = next_state
                self.pred_traj.append(self.state)
                if check_repeat(self.pred_traj):
                    status = NavStatus.repeat
                else:
                    status = NavStatus.in_progress
            else:
                status = NavStatus.invalid_action
        if status == NavStatus.success:
            reward = 1.0
        elif status in [NavStatus.invalid_action, NavStatus.false_complete]:
            reward = -1.0
        else:
            reward = 0
        done = status != NavStatus.in_progress
        if not done and self.count_steps == self.max_steps - 1:
            status = NavStatus.max_step_exceeded
        return self.obsv(), reward, done, {'status': status}

    def _init_planner(self) -> GridPlanner:
        grid = self.gridmap.fill()
        return EgoGridPlanner(self.meta, grid, allow_backward=self.allow_backward) if self.ego \
            else GridMapPlanner(self.meta, grid)

    def _reset_traj(self):
        trajs = []
        planner = None
        while not trajs:
            planner = self._init_planner()
            trajs = planner.sample_trajectories(self.min_traj_len, n_trajs=1)
        return trajs[0], planner

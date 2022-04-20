from copy import deepcopy
from typing import Tuple, Any

import numpy as np

from core.domains.gridworld.actions import GridDirs
from core.domains.gridworld.planner.gridmap_planner import GridMapPlanner, GridMDPMeta
from core.mdp.actions import OmniActionSet
from core.mdp.meta import MDPMeta
from core.utils.geometry_utils import sample_free_space, get_world_coord, rotation_from_camera_normal
from core.utils.env_utils import update_env
from core.domains.miniworld.actions import MiniWorldActionSet
from gym_miniworld.math import intersect_circle_segs
from gym_miniworld.miniworld import MiniWorldEnv


class MiniWorldTrajState:
    def __init__(self, env: MiniWorldEnv, parent=None, is_terminal=False):
        self._env = env
        self.pos = env.agent.pos  # immutable
        self.dir = env.agent.dir  # immutable
        self.step_count = env.step_count  # immutable
        self.parent = parent
        # self.action = MiniWorldActionSet().done if is_terminal else None
        self.is_terminal = is_terminal
        self.rgb = env.render_obs()
        self.depth = env.render_depth()
        self.world_coords = self.get_world_coords(self.depth)

    def get_full_view(self, n_samples_per_pixel=1):
        env = self.env()
        rgbs, depths, world_coords, free_spaces = [], [], [], []
        turn_step_deg = env.params.params['turn_step'].default
        for _ in range(360 // turn_step_deg):
            rgbs.append(env.render_obs())
            depth = env.render_depth()
            depths.append(depth)
            coords = self.get_world_coords(depth)
            world_coords.append(coords)
            free = sample_free_space(coords, env.agent.cam_pos, n_samples_per_pixel=n_samples_per_pixel)
            free_spaces.append(free)
            env.step(MiniWorldEnv.Actions.turn_right)
        return rgbs, depths, world_coords, free_spaces

    def env(self):
        env = self._env
        env.agent.pos = deepcopy(self.pos)
        env.agent.dir = deepcopy(self.dir)
        env.step_count = self.step_count
        return env

    def __repr__(self):
        return f"{self.pos} {self.dir}"

    def point_cloud(self):
        return np.concatenate([self.world_coords, self.rgb], axis=-1).reshape((-1, 6))

    def sample_free_space(self, world_coords=None, n_samples_per_pixel=1):
        if world_coords is None: world_coords = self.world_coords
        return sample_free_space(world_coords, self._env.agent.cam_pos, n_samples_per_pixel=n_samples_per_pixel)

    def get_world_coords(self, depth):
        env = self._env
        R = rotation_from_camera_normal(env.agent.cam_dir)
        t = env.agent.cam_pos
        depth = depth.squeeze(2)
        fov_y = env.agent.cam_fov_y
        hfov = np.deg2rad(fov_y)
        return get_world_coord(depth, R, t, hfov)


class MiniWorldMDPMeta(MDPMeta):
    def __init__(self, map_bbox: tuple, map_res: tuple, ori_res):
        self.dirs = dirs = GridDirs(ori_res == 4) if ori_res in [4, 8] else None
        self.map_bbox = map_bbox
        self.map_res = map_res
        self.ori_res = ori_res
        super(MiniWorldMDPMeta, self).__init__(MiniWorldActionSet() if ori_res else OmniActionSet(ori_res),
                                               (ori_res, *map_res) if ori_res else map_res, dirs)

    def state_to_index(self, state: MiniWorldTrajState):
        pose = self.state_to_pose(state)
        return self.pose_to_index(pose)

    def state_index_to_grid_index(self, state_index):
        return state_index[1:] if self.ori_res else state_index

    def _vec_to_dir(self, h, w):
        if h == 0: return 0 if w > 0 else -np.pi
        if h > 0: return np.arctan(w / h) - np.pi / 2
        return np.arctan(w / h) + np.pi / 2

    def pos_rescale(self, pos):
        w, _, h = pos
        h1, w1, h2, w2 = self.map_bbox
        assert h1 <= h < h2, f"h dimension {h} out of range [{h1}, {h2})"
        assert w1 <= w < w2, f"w dimension {w} out of range [{w1}, {w2})"
        h_size, w_size = self.map_res
        h_rescale = (h - h1) / (h2 - h1) * h_size
        w_rescale = (w - w1) / (w2 - w1) * w_size
        return h_rescale, w_rescale

    def pos_to_grid_index(self, pos):
        h_rescale, w_rescale = self.pos_rescale(pos)
        return int(h_rescale), int(w_rescale)

    def dir_to_ori_index(self, d):
        return int((self.ori_res * d / (2 * np.pi) + 0.5) % self.ori_res)

    def grid_index_to_pos(self, grid_index):
        h1, w1, h2, w2 = self.map_bbox
        h_size, w_size = self.map_res
        h_ind, w_ind = grid_index
        return w_ind / w_size * (w2 - w1) + w1, 0, h_ind / h_size * (h2 - h1) + h1

    def state_to_pose(self, state):
        w, _, h = state.pos
        return state.dir, h, w

    def pose_to_index(self, pose):
        d, h, w = pose
        h, w = self.pos_to_grid_index((w, 0, h))
        o = self.dir_to_ori_index(d)
        return (o, h, w) if self.ori_res else (h, w)


def update_miniworld_env(meta: MiniWorldMDPMeta, env: MiniWorldEnv, action) -> Tuple[Any, float, bool, Any]:
    turn_step_deg = env.params.params['turn_step'].default
    turn_step = np.deg2rad(turn_step_deg)
    return update_env(meta, env, action, curr_dir=env.agent.dir, turn_step=turn_step, action_set=MiniWorldActionSet())


class MiniWorldPlanner:
    def __init__(self, meta: MiniWorldMDPMeta, env: MiniWorldEnv, *, costmap_margin=2, costmap_coeff=10, radius=0.2):
        self.meta = meta
        self.env = env
        self.radius = radius
        self.gridmap_planner = GridMapPlanner(GridMDPMeta(meta.map_res), self.grid, costmap_coeff=costmap_coeff,
                                              costmap_margin=costmap_margin)

        target = env.box.pos
        self.source = MiniWorldTrajState(self.env, None)
        self.target_index = meta.pos_to_grid_index(target)

        self.grid_cost, self.best_trans, _ = self.gridmap_planner.get_transition_tree(self.target_index,
                                                                                      is_root_target=True)
        costs = np.ones(meta.map_res, dtype=np.float) * np.array(list(self.grid_cost.values())).max() + 1
        for state, cost in self.grid_cost.items():
            costs[state] = cost
        self.costs = costs

        turn_step_deg = env.params.params['turn_step'].default
        self.turn_step = np.deg2rad(turn_step_deg)

    def transition(self, curr_state: MiniWorldTrajState, action):
        """
        :param curr_state:
        :param action:
        :return: next_state
        """
        env = curr_state.env()
        if self.meta.ori_res:
            _, reward, done, _ = env.step(action)
        else:
            _, reward, done, _ = update_miniworld_env(self.meta, env, action)
            # obs, reward, done, info = env.step(action)
        if reward == 0 and (np.array_equal(env.agent.pos, curr_state.pos) and env.agent.dir == curr_state.dir):
            return None  # Wrong move
        next_state = MiniWorldTrajState(env, self, is_terminal=done and reward > 0)
        return next_state

    def _get_transitions(self, curr_state):
        """
        :param curr_state:
        :param reverse: if True, return prev states rather than next states
        :return: list of (action, next_state, cost) tuples, (action, prev_state, cost) if reversed
        """
        if self.meta.ori_res:
            meta = self.meta
            states = []
            actions = []

            curr_grid_index = meta.pos_to_grid_index(curr_state.pos)
            future_grid_index = self._n_step_lookahead(curr_grid_index)

            d = meta._vec_to_dir(future_grid_index[0] - curr_grid_index[0], future_grid_index[1] - curr_grid_index[1])
            angle_diff = (d - curr_state.dir) % (2 * np.pi)
            if angle_diff < np.pi:
                n_turns = int(angle_diff / self.turn_step + 0.5)
                action = meta.actions.turn_left
            else:
                n_turns = int((2 * np.pi - angle_diff) / self.turn_step + 0.5)
                action = meta.actions.turn_right
            for _ in range(int(n_turns)):
                curr_state = self.transition(curr_state, action)
                assert curr_state is not None
                states.append(curr_state)
                actions.append(action)
                if curr_state.is_terminal: return states, actions

            action = meta.actions.move_forward
            curr_state = self.transition(curr_state, action)
            states.append(curr_state)
            actions.append(action)

            return states, actions
        else:
            curr_grid_index = self.meta.pos_to_grid_index(curr_state.pos)
            future_grid_index = self._n_step_lookahead(curr_grid_index)
            d = self.meta._vec_to_dir(future_grid_index[0] - curr_grid_index[0], future_grid_index[1] - curr_grid_index[1])
            action = self.meta.actions.dir_to_action(d)
            return [self.transition(curr_state, action)], [action]

    def get_trajectory(self):
        """
        :param source: source state
        :param target: target state
        :return: list of tuples (state, action) along trajectory. last pair is (target, done)
        """
        state = self.source
        traj_states = [state]
        traj_actions = []
        while not state.is_terminal:
            states, actions = self._get_transitions(state)
            state = states[-1]
            if state is None or self.best_trans[self.meta.pos_to_grid_index(state.pos)] is None:
                return []
            traj_states += states
            traj_actions += actions
        traj_actions.append(self.meta.actions.done)
        return list(zip(traj_states, traj_actions, traj_states[1:] + [traj_states[-1]]))

    def grid_trajectory(self):
        """
        :return: list of tuples (state, action) along trajectory. last pair is (target, None)
        """
        curr_state_index = self.meta.pos_to_grid_index(self.source.pos)
        traj = [curr_state_index]
        while not curr_state_index == self.target_index:
            curr_state_index, _ = self.best_trans[curr_state_index]
            traj.append(curr_state_index)
        return traj

    def intersect(self, pos):
        """
        Check if an entity intersects with the world
        """
        # Ignore the Y position
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Check for intersection with walls
        if intersect_circle_segs(pos, self.radius, self.env.wall_segs):
            return True

    def _n_step_lookahead(self, grid_index, n=2):
        for _ in range(n):
            next_pair = self.best_trans[grid_index]
            if next_pair is None: break
            grid_index, _ = next_pair
        return grid_index

    @property
    def grid(self):
        grid = np.ones(self.meta.map_res)
        for grid_index in zip(*np.where(grid)):
            pos = self.meta.grid_index_to_pos(grid_index)
            if not self.intersect(pos):
                grid[grid_index] = 0
        return grid

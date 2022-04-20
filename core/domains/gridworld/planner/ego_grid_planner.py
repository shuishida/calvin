import sys

import numpy as np

sys.path.append('.')

from core.mdp.meta import MDPMeta
from core.mdp.actions import EgoActionSetBase, EgoActionSet
from core.domains.gridworld.actions import GridDirs
from core.mdp.grid_planner import GridPlanner


def get_new_state(state, action, dirs: GridDirs, reverse=False, action_set: EgoActionSetBase = EgoActionSet,
                  step_size=1):
    (dir_x, dir_y), x, y = state
    if action in [action_set.turn_right, action_set.turn_left]:
        new_dir_x, new_dir_y = dirs.rotate((dir_x, dir_y), reverse ^ (action == action_set.turn_right))
        return (new_dir_x, new_dir_y), x, y
    elif action == action_set.done:
        return None
    if reverse ^ (action == action_set.move_backward):
        return (dir_x, dir_y), x - dir_x * step_size, y - dir_y * step_size
    else:
        expected_action = action_set.move_backward if reverse else action_set.move_forward
        assert action == expected_action, f"Expected action {expected_action}, got {action}"
        return (dir_x, dir_y), x + dir_x * step_size, y + dir_y * step_size


class EgoGridMDPMeta(MDPMeta):
    def __init__(self, grid_shape: tuple, four_way=False):
        dirs = GridDirs(four_way)
        super(EgoGridMDPMeta, self).__init__(EgoActionSet(), state_shape=(len(dirs), *grid_shape), dirs=dirs)

    def state_to_index(self, state):
        d, x, y = state
        return self.dirs.index(d), int(x), int(y)

    def state_index_to_grid_index(self, state_index):
        return state_index[1:]

    def state_to_pose(self, state):
        d, x, y = state
        return self.dirs.degrees(d), x, y


class EgoGridPlanner(GridPlanner):
    def __init__(self, meta: EgoGridMDPMeta, grid, allow_backward=False):
        states = list((d, x, y) for (x, y) in zip(*np.where(~grid.astype(bool))) for d in meta.dirs)
        super(EgoGridPlanner, self).__init__(meta, states, grid)
        self.allow_backward = allow_backward

    def get_motion(self, state_index, next_state_index, i):
        if i == 0: return next_state_index[i]
        return next_state_index[i] - state_index[i]

    def is_valid_state(self, state):
        d, row, col = state
        n_row, n_col = self.grid.shape
        return (0 <= row < n_row) and (0 <= col < n_col) and (self.grid[row, col] == 0)

    def transition(self, curr_state, action, reverse=False):
        """
        :param curr_state:
        :param action:
        :param reverse: if reversed, give previous state instead of next state
        :return: list of (next_state, action, cost) tuples, (prev_state, action, cost) if reversed
        """
        meta = self.meta
        if action == meta.actions.done: return None
        if action == meta.actions.move_backward and not self.allow_backward: return None
        _, x, y = curr_state
        new_state = get_new_state(curr_state, action, dirs=meta.dirs, reverse=reverse, action_set=meta.actions)
        new_dir, new_x, new_y = new_state
        cond = self.is_valid_state(new_state)
        if x != new_x and y != new_y:
            cond = cond and (self.is_valid_state((new_dir, x, new_y)) or self.is_valid_state((new_dir, new_x, y)))
        if cond:
            return new_state, action, 1
        return None

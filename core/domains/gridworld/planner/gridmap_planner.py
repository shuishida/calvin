import numpy as np
import sys
import torch
import torch.nn.functional as F


sys.path.append('.')

from core.domains.gridworld.actions import GridActionSet
from core.mdp.meta import MDPMeta
from core.mdp.grid_planner import GridPlanner


class GridMDPMeta(MDPMeta):
    def __init__(self, state_shape: tuple, four_way=False):
        super(GridMDPMeta, self).__init__(GridActionSet(four_way=four_way), state_shape)

    def state_to_index(self, state):
        return tuple(map(int, state))

    def state_index_to_grid_index(self, state_index):
        return state_index


class GridMapPlanner(GridPlanner):
    def __init__(self, meta: GridMDPMeta, grid, costmap_margin=2, costmap_coeff=0.0):
        states = list(zip(*np.where(~grid.astype(bool))))
        super(GridMapPlanner, self).__init__(meta, states, grid)
        self.costmap = self.create_costmap(costmap_margin, costmap_coeff)

    def create_costmap(self, p, a):
        """
        :param p: kernel padding size
        :param a: coefficient
        :return: costmap
        """
        if not (p > 0 and a > 0): return np.zeros(self.meta.state_shape)
        grid = torch.from_numpy(self.grid).unsqueeze(0).unsqueeze(0).float()
        weight_1d = p + 1 - torch.abs(torch.arange(2 * p + 1) - p)
        weight_2d = weight_1d * weight_1d.view(-1, 1)
        weight_2d = weight_2d ** 2
        weights = weight_2d.unsqueeze(0).unsqueeze(0).float()
        costmap = a * F.conv2d(grid, weights, padding=p).squeeze(0).squeeze(0).data.numpy()
        return costmap

    def is_valid_state(self, state):
        row, col = state
        n_row, n_col = self.grid.shape
        return (0 <= row < n_row) and (0 <= col < n_col) and (self.grid[state] == 0)

    def get_state_cost(self, state):
        return self.costmap[state]

    def transition(self, curr_state, action, reverse=False):
        """
        :param curr_state:
        :param action:
        :param reverse: if reversed, give previous state instead of next state
        :return: (next_state, action, cost) tuple, (prev_state, action, cost) if reversed
        """
        if action == self.meta.actions.done: return None
        x, y = curr_state
        move_x, move_y = action
        if reverse:
            move_x, move_y = -move_x, -move_y
        new_state = (x + move_x, y + move_y)
        cond = self.is_valid_state(new_state)
        if move_x != 0 and move_y != 0:
            cond = cond and (self.is_valid_state((x, y + move_y)) or self.is_valid_state((x + move_x, y)))
        if cond:
            cost = (move_x ** 2 + move_y ** 2) ** 0.5
            return new_state, action, cost
        return None

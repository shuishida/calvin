import numpy as np
from math import pi

from core.domains.gridworld.agent_view.agent_view import AgentView


class AgentMap:
    def __init__(self, grid: np.ndarray, target=None, view_range=None,
                 view_angle=2 * pi, target_known=False):
        self.grid = grid
        self.target = target
        self.get_view = AgentView(view_range=view_range, view_angle=view_angle) if view_range else None
        self.view = self.init_view()
        self.target_known = target_known

    def init_view(self):
        if self.get_view:
            return np.zeros((self.get_view.N_CHANNELS, *self.grid.shape))
        else:
            return np.expand_dims(self.grid, 0)

    def embed_target(self, visible=None, target=None):
        # embed target position in map
        embed = np.zeros((1, *self.grid.shape))
        if visible is None: visible = np.ones_like(self.grid)
        if target is None: target = self.target
        if target and visible[target] or self.target_known:
            embed[0, target[0], target[1]] = 1
        return embed

    def get_visibility(self):
        if self.get_view:
            return self.view.max(axis=0).astype(bool).astype(int)
        else:
            return np.ones_like(self.grid)

    def get_obsv(self):
        visibility = self.get_visibility()
        return np.concatenate([self.view, self.embed_target(visibility)]), visibility

    def update(self, pose):
        if self.get_view:
            if self.get_view.view_angle != 2 * pi:
                assert len(pose) == 3, "orientation of the agent not defined"
            view = self.get_view.glob(self.grid, *pose).astype(bool)
            # swap 0th and 1st channels so that 0th channel: obstacles, 1st channel: clear space
            view = view[[1, 0], ...]
            # take an OR operation and update view
            self.view = (self.view.astype(bool) | view).astype(int)
        return self.get_obsv()

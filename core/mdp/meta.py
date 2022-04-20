import numpy as np

from core.mdp.actions import ActionSetBase


class EnvMeta:
    def sample_actions(self, n: int = None):
        raise NotImplementedError


class MDPMeta(EnvMeta):
    def __init__(self, action_set: ActionSetBase, state_shape: tuple, dirs=None):
        self.actions = action_set
        self.state_shape = state_shape
        self.dirs = dirs

    def sample_actions(self, n: int = None):
        return [self.actions[i] for i in np.random.permutation(len(self.actions))[:n]]

    def action_to_index(self, action):
        return self.actions.index(action)

    def action_from_index(self, action_index):
        return self.actions[action_index]

    def state_to_grid_index(self, state):
        return self.state_index_to_grid_index(self.state_to_index(state))

    def state_to_index(self, state):
        raise NotImplementedError

    def state_index_to_grid_index(self, state_index):
        raise NotImplementedError

    def state_to_pose(self, state):
        raise NotImplementedError

    def pose_to_index(self, pose):
        raise NotImplementedError

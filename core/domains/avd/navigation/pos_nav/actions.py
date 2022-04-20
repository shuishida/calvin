import numpy as np
from enum import Enum

from core.domains.gridworld.actions import GridDirs
from core.mdp.actions import ActionSetBase

AVD_ACTIONS_8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (0, 0)]
AVD_ACTIONS_4 = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]
AVD_COMPLETE = (0, 0)


class DirAction(Enum):
    E = 0
    NE = 1
    N = 2
    NW = 3
    W = 4
    SW = 5
    S = 6
    SE = 7
    COMPLETE = -1


class AVDPosActionSet(ActionSetBase):
    def __init__(self):
        self.dirs = GridDirs(four_way=False)
        super(AVDPosActionSet, self).__init__(DirAction)

    @property
    def done(self):
        return DirAction.COMPLETE


def action_to_dir(action: DirAction):
    angle = action_to_angle(action)
    return np.array([np.cos(angle), 0, np.sin(angle)])


def action_to_move(action: DirAction, step_size: float):
    angle = action_to_angle(action)
    return np.array([np.cos(angle), 0, np.sin(angle)]) * step_size


def action_to_angle(action: DirAction):
    if action == DirAction.COMPLETE: raise Exception("This action is a COMPLETE action")
    return np.pi / 4 * action.value


def action_from_angle(angle):
    angle = angle % (2 * np.pi)
    value = int(angle / (np.pi / 4) + 0.5) % 8
    return DirAction(value)


def get_reverse_action(action):
    if action == DirAction.COMPLETE: return action
    return DirAction((action.value + 4) % 8)

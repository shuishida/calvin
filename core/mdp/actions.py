import numpy as np
from enum import Enum


class ActionSetBase(list):
    @property
    def done(self):
        pass

    @property
    def done_index(self):
        return self.index(self.done)


class EgoActionSetBase(ActionSetBase):
    @property
    def move_forward(self):
        pass

    @property
    def move_backward(self):
        pass

    @property
    def turn_right(self):
        pass

    @property
    def turn_left(self):
        pass

    def nav_actions(self):
        return [self.move_forward, self.move_backward, self.turn_right, self.turn_left]


class EgoAction(Enum):
    move_forward = 0
    move_backward = 1
    turn_right = 2
    turn_left = 3
    done = -1


class EgoActionSet(EgoActionSetBase):
    def __init__(self):
        self.actions = EgoAction
        super(EgoActionSet, self).__init__(list(self.actions))

    @property
    def done(self):
        return self.actions.done

    @property
    def move_forward(self):
        return self.actions.move_forward

    @property
    def move_backward(self):
        return self.actions.move_backward

    @property
    def turn_right(self):
        return self.actions.turn_right

    @property
    def turn_left(self):
        return self.actions.turn_left


class OmniActionSet(ActionSetBase):
    def __init__(self, ori_res: int):
        self._done = ori_res
        self.ori_res = ori_res
        super(OmniActionSet, self).__init__(list(range(ori_res + 1)))

    @property
    def done(self):
        return self._done

    def assert_action(self, action, exclude_done=False):
        assert isinstance(action, int)
        assert 0 <= action <= self.ori_res - int(exclude_done)

    def degrees(self, action: int):
        self.assert_action(action, exclude_done=True)
        return action * 360 / self.ori_res

    def dir(self, action: int):
        self.assert_action(action, exclude_done=True)
        return action * 2 * np.pi / self.ori_res

    def dir_to_action(self, d: float):
        return self[int((self.ori_res * d / (2 * np.pi) + 0.5) % self.ori_res)]
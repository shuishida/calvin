from gym_miniworld.miniworld import MiniWorldEnv
from core.mdp.actions import EgoActionSetBase


class MiniWorldActionSet(EgoActionSetBase):
    def __init__(self):
        self.actions = MiniWorldEnv.Actions
        super(MiniWorldActionSet, self).__init__(list(self.actions))

    @property
    def done(self):
        return self.actions.done

    @property
    def move_forward(self):
        return self.actions.move_forward

    @property
    def move_backward(self):
        return self.actions.move_back

    @property
    def turn_right(self):
        return self.actions.turn_right

    @property
    def turn_left(self):
        return self.actions.turn_left

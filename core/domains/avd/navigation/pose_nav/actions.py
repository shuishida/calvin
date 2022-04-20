from core.domains.avd.dataset.const import AVDMove
from core.mdp.actions import EgoActionSetBase


class AVDPoseActionSet(EgoActionSetBase):
    def __init__(self):
        self.actions = AVDMove
        self._done = "DONE"
        super(AVDPoseActionSet, self).__init__(list(self.actions) + [self._done])

    @property
    def done(self):
        return self._done

    @property
    def move_forward(self):
        return self.actions.forward

    @property
    def move_backward(self):
        return self.actions.backward

    @property
    def turn_right(self):
        return self.actions.rotate_cw

    @property
    def turn_left(self):
        return self.actions.rotate_ccw

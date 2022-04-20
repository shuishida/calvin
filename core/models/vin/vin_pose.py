import torch
import torch.nn as nn

from core.models.projection.point_cloud_vin_base import PointCloudVINBase
from core.models.vin.vin import VIN
from core.models.vin.vin_utils import pose_acc, pose_extract_state_values, action_value_loss
from einops import rearrange


"""
Value Iteration Network
"""


class VINPose(VIN):
    def __init__(self, *, ori_res=None, **config):
        self.k_ori = ori_res
        super(VINPose, self).__init__(**config)

    def _init(self):
        self.r2q = nn.Conv2d(in_channels=1, out_channels=self.l_q * self.k_ori,
                             kernel_size=(self.k_sz, self.k_sz), stride=1, padding=self.p, bias=True)
        self.v2q = nn.Conv3d(in_channels=1, out_channels=self.l_q * self.k_ori,
                             kernel_size=(self.k_ori, self.k_sz, self.k_sz), stride=1, padding=(0, self.p, self.p),
                             bias=True)
        self.policy = nn.Conv3d(in_channels=self.l_q, out_channels=len(self.actions),
                                kernel_size=(1, 1, 1), stride=1, padding=0,
                                bias=False) if self.use_policy_net else None

    def __repr__(self):
        return f"{super().__repr__()}_ori_{self.k_ori}"

    def eval_q(self, r, v=None):
        q = self.r2q(r).unsqueeze(2)
        if v is not None:
            q += self.gamma * self.v2q(v)
        return rearrange(q, "b (a o1) () x y -> b a o1 x y", o1=self.k_ori)

    def metrics(self, q=None, best_action_maps=None, loss_weights=None, **kwargs) -> dict:
        if q is None or best_action_maps is None or loss_weights is None: return {}
        return {'acc': pose_acc(q, best_action_maps, loss_weights)}

    def extract_state_q(self, q, state):
        return pose_extract_state_values(q, state)

    def loss(self, q=None, **kwargs):
        loss = action_value_loss(q, discount=self.discount, sparse=self.sparse, **kwargs)
        return loss, {}


class VINPoseNav(PointCloudVINBase, VINPose):
    pass

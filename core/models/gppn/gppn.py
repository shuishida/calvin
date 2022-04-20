import torch
import torch.nn as nn

from einops import rearrange

from core.models.projection.point_cloud_vin_base import PointCloudVINBase
from core.models.vin.vin_base import VINBase
from core.models.vin.vin_utils import pos_acc, pos_extract_state_values, \
    pose_acc, \
    pose_extract_state_values, action_value_loss

"""
Gated Path Planning Networks
"""


class GPPN(VINBase):
    def __init__(self, *, k_sz=None, l_h=None, ori_res=None, **config):
        super(GPPN, self).__init__(**config)

        self.l_h = l_h
        self.k_sz = k_sz
        self.p = k_sz // 2
        self.k_ori = ori_res

        self.hid = nn.Conv2d(in_channels=self.l_i, out_channels=self.l_h,
                             kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.h0 = nn.Conv2d(in_channels=self.l_h, out_channels=self.l_h,
                            kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.c0 = nn.Conv2d(in_channels=self.l_h, out_channels=self.l_h,
                            kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.conv = nn.Conv2d(in_channels=self.l_h, out_channels=1,
                              kernel_size=(self.k_sz, self.k_sz), stride=1, padding=self.p, bias=True)

        self.lstm = nn.LSTMCell(1, self.l_h)

        self.policy = nn.Conv2d(in_channels=self.l_h, out_channels=len(self.actions) * ori_res if ori_res else len(self.actions),
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)

    def __repr__(self):
        return f"{super().__repr__()}_ksz_{self.k_sz}_i_{self.l_i}_h_{self.l_h}"

    def forward(self, feature_map=None, k=None, prev_v=None, **kwargs):
        """
        :param feature_map: (batch_sz, imsize, imsize)
        :param k: number of iterations. If None, use config.k
        :param prev_v: previously evaluated v (if it exists)
        :return: logits and softmaxed logits
        """
        batch_sz, l_i, map_x, map_y = feature_map.size()

        hid = self.hid(feature_map)
        h0 = rearrange(self.h0(hid), "b h x y -> (b x y) h")
        c0 = rearrange(self.c0(hid), "b h x y -> (b x y) h")

        last_h, last_c = h0, c0
        for _ in range(0, self.k - 1):
            h_map = rearrange(last_h, "(b x y) h -> b h x y", x=map_x, y=map_y)
            inp = rearrange(self.conv(h_map), "b () x y -> (b x y) ()")
            last_h, last_c = self.lstm(inp, (last_h, last_c))

        hk = rearrange(last_h, "(b x y) h -> b h x y", x=map_x, y=map_y)
        q = self.policy(hk)
        if self.k_ori: q = rearrange(q, "b (a o) x y -> b a o x y", o=self.k_ori)
        v, _ = torch.max(q, dim=1, keepdim=True)

        return {"q": q, "v": v}

    def metrics(self, q=None, best_action_maps=None, loss_weights=None, **kwargs) -> dict:
        if q is None or best_action_maps is None or loss_weights is None: return {}
        return {'acc': pos_acc(q, best_action_maps, loss_weights)}

    def extract_state_q(self, q, state):
        return pos_extract_state_values(q, state)

    def loss(self, q=None, **kwargs):
        loss = action_value_loss(q, discount=self.discount, sparse=self.sparse, **kwargs)
        return loss, {}


class GPPNNav(PointCloudVINBase, GPPN):
    pass


class GPPNPose(GPPN):
    def metrics(self, q=None, best_action_maps=None, loss_weights=None, **kwargs) -> dict:
        if q is None or best_action_maps is None or loss_weights is None: return {}
        return {'acc': pose_acc(q, best_action_maps, loss_weights)}

    def extract_state_q(self, q, state):
        return pose_extract_state_values(q, state)

    def loss(self, q=None, **kwargs):
        loss = action_value_loss(q, discount=self.discount, sparse=self.sparse, **kwargs)
        return loss, {}


class GPPNPoseNav(PointCloudVINBase, GPPNPose):
    pass

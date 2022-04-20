import torch
import torch.nn as nn

from core.models.projection.point_cloud_vin_base import PointCloudVINBase
from core.models.vin.vin_base import VINBase
from core.models.vin.vin_utils import pos_acc, pos_extract_state_values, action_value_loss

"""
Value Iteration Network
"""


class VIN(VINBase):
    def __init__(self, *, k_sz=None, l_h=None, l_q=None, use_policy_net=False, **config):
        super(VIN, self).__init__(**config)

        self.l_h = l_h
        self.l_q = l_q
        self.k_sz = k_sz
        self.p = k_sz // 2

        self.reward_net = nn.Sequential(
            nn.Conv2d(in_channels=self.l_i, out_channels=l_h,
                      kernel_size=(k_sz, k_sz), stride=1, padding=self.p, bias=True),
            nn.Conv2d(in_channels=l_h, out_channels=1,
                      kernel_size=(1, 1), stride=1, padding=0, bias=False)
        )

        self.use_policy_net = use_policy_net

        self._init()

    def _init(self):
        self.r2q = nn.Conv2d(in_channels=1, out_channels=self.l_q,
                             kernel_size=(self.k_sz, self.k_sz), stride=1, padding=self.p, bias=True)
        self.v2q = nn.Conv2d(in_channels=1, out_channels=self.l_q,
                             kernel_size=(self.k_sz, self.k_sz), stride=1, padding=self.p, bias=True)
        self.policy = nn.Conv2d(in_channels=self.l_q, out_channels=len(self.actions),
                                kernel_size=(1, 1), stride=1, padding=0, bias=False) if self.use_policy_net else None

    def __repr__(self):
        return f"{super().__repr__()}_i_{self.l_i}_h_{self.l_h}_q_{self.l_q}{'_pn' if self.policy else ''}"

    def eval_q(self, r, v=None):
        q = self.r2q(r)
        if v is not None:
            q += self.gamma * self.v2q(v)
        return q

    def _forward(self, feature_map=None, k=None, prev_v=None, **kwargs):
        """
        :param feature_map: (batch_sz, imsize, imsize)
        :param k: number of iterations. If None, use config.k
        :param prev_v: previously evaluated v (if it exists)
        :return: logits and softmaxed logits
        """
        r = self.reward_net(feature_map)  # Reward
        q = self.eval_q(r, prev_v)
        v, _ = torch.max(q, dim=1, keepdim=True)

        # Update q and v values
        if k is None: k = self.k
        for i in range(k):
            q = self.eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        if self.policy:
            q = self.policy(q)

        return {"q": q, "v": v, "r": r, "r2q": self.r2q.weight, "v2q": self.v2q.weight}

    def metrics(self, q=None, best_action_maps=None, loss_weights=None, **kwargs) -> dict:
        if q is None or best_action_maps is None or loss_weights is None: return {}
        return {'acc': pos_acc(q, best_action_maps, loss_weights)}

    def extract_state_q(self, q, state):
        return pos_extract_state_values(q, state)

    def loss(self, q=None, **kwargs):
        loss = action_value_loss(q, discount=self.discount, sparse=self.sparse, **kwargs)
        return loss, {}


class VINPosNav(PointCloudVINBase, VIN):
    pass


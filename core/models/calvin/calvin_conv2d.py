import torch
import torch.nn.functional as F
from einops import reduce, rearrange
from torch.nn import Parameter
import torch.nn as nn

from core.models.calvin.calvin_base import CALVINBase
from core.models.projection.point_cloud_vin_base import PointCloudVINBase
from core.models.vin.vin_utils import pos_acc, pos_extract_state_values, action_value_loss, pos_motion_loss, \
    make_conv_layers


class CALVINConv2d(CALVINBase):
    def __init__(self, *, l_h=None, k_sz=None, motion_scale=10, w_loss_p=None, n_layers=None, **config):
        super(CALVINConv2d, self).__init__(**config)

        self.kx = k_sz
        self.ky = k_sz
        self.l_h = l_h
        self.motion_scale = motion_scale

        self.w_loss_p = w_loss_p

        kernel_c_x = (self.kx - 1) // 2
        kernel_c_y = (self.ky - 1) // 2
        self.padding = (kernel_c_x, kernel_c_y)

        self.aa_net = make_conv_layers(l_i=self.l_i, l_h=l_h, l_o=len(self.actions) + 1,
                                       kx=self.kx, ky=self.ky, n_layers=n_layers, dropout=self.dropout)

        self.warning_given = False

        self.r_failure = Parameter(torch.zeros((1,), device=self.device), requires_grad=True)

        weight_shape = (len(self.actions), 1, self.kx, self.ky)

        self.w_motion = Parameter(torch.randn(weight_shape, device=self.device), requires_grad=True)
        self.r_motion = Parameter(torch.zeros(weight_shape, device=self.device), requires_grad=True)

    def __repr__(self):
        return f"{super().__repr__()}_i_{self.l_i}_h_{self.l_h}"

    def get_w_motion(self):
        return self.w_motion * self.motion_scale

    def get_motion_model(self):
        w_motion = self.get_w_motion()
        motion_flatten = w_motion.view(len(self.actions), -1)
        motion_model = F.softmax(motion_flatten, dim=-1).view(w_motion.shape)
        return motion_model

    def get_available_actions(self, input_view, motion_model, target_map):
        aa_out = self.aa_net(input_view)
        aa_logit, aa_thresh = aa_out[:, :-1], aa_out[:, -1:]
        aa = torch.sigmoid(aa_logit - aa_thresh)
        return aa, aa_logit, aa_thresh, None, None

    def get_reward_function(self, feature_map, available_actions, motion_model):
        """
        :param available_actions: (batch_sz, n_actions, map_x, map_y)
        :param reward_map: (batch_sz, 1, map_x, map_y)
        :return: reward function R(s, a): (batch_sz, n_actions, map_x, map_y)
        """
        # penalty of taking an invalid action
        reward = - F.softplus(self.r_failure) * (1 - available_actions)
        # reward[:, self.actions.done_index, :, :] = reward_map.squeeze(1)
        motion_reward = reduce(motion_model * self.r_motion, "a () kx ky -> () a () ()", "sum", kx=self.kx, ky=self.ky)
        return reward + available_actions * motion_reward

    def eval_q(self, available_actions, motion_model, reward, value=None):
        q = reward.clone()
        if value is not None:
            v_next = F.conv2d(value, motion_model, stride=1, padding=self.padding)
            v_next[:, self.actions.done_index, :, :] = 0  # termination states
            q = q + self.gamma * available_actions * v_next
        return q

    def loss(self, q=None, aa_logit=None, **kwargs):
        loss_qs = action_value_loss(q, discount=self.discount, sparse=self.sparse, **kwargs)
        loss_aa = action_value_loss(aa_logit, discount=self.discount, sparse=self.sparse, **kwargs)
        loss_motion = pos_motion_loss(self.get_w_motion(), **kwargs)

        # print(loss_qs, loss_aa, loss_motion)
        loss = loss_qs + loss_aa + loss_motion * self.w_loss_p
        return loss, {'loss_qs': loss_qs, 'loss_aa': loss_aa, 'loss_motion': loss_motion}

    def metrics(self, q=None, best_action_maps=None, loss_weights=None, **kwargs) -> dict:
        if q is None or best_action_maps is None or loss_weights is None: return {}
        return {'acc': pos_acc(q, best_action_maps, loss_weights)}

    def extract_state_q(self, q, state):
        return pos_extract_state_values(q, state)


class CALVINPosNav(PointCloudVINBase, CALVINConv2d):
    pass

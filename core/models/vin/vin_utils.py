import torch
from einops import rearrange
from torch.nn import functional as F
import torch.nn as nn


def pos_acc(q=None, best_action_maps=None, loss_weights=None, **kwargs):
    pred_action_labels = q.argmax(dim=1)
    pred_action_labels[loss_weights == 0] = -1
    batch, mapx, mapy = torch.where(pred_action_labels != -1)
    actions = pred_action_labels[batch, mapx, mapy]
    return best_action_maps[batch, actions, mapx, mapy].float().mean().item()


def pos_extract_state_values(values, state, index=None):
    if index is None: index = torch.arange(len(state))
    return values[index, :, state[:, 0], state[:, 1]]  # (batch_sz, n_actions)


def pose_acc(q=None, best_action_maps=None, loss_weights=None, **kwargs):
    pred_action_labels = q.argmax(dim=1)
    pred_action_labels[loss_weights == 0] = -1
    batch, ori, mapx, mapy = torch.where(pred_action_labels != -1)
    actions = pred_action_labels[batch, ori, mapx, mapy]
    return best_action_maps[batch, actions, ori, mapx, mapy].float().mean().item()


def pose_extract_state_values(values, state, index=None):
    if index is None: index = torch.arange(len(state))
    return values[index, :, state[:, 0], state[:, 1], state[:, 2]]  # (batch_sz, n_actions)


def action_value_loss(action_values, *, curr_poses=None, poses=None, curr_actions=None, actions=None,
                      discount=None, lens=None, index=None, step=None, sparse=None, **kwargs):
    assert isinstance(discount, float)
    assert isinstance(sparse, bool)
    curr_poses = curr_poses.long()
    poses = poses.long()
    batch_sz, pose_size = curr_poses.size()
    discounts = discount ** (lens[index] - step - 1) if discount else 1
    if pose_size == 2: extract_state_value = pos_extract_state_values
    elif pose_size == 3: extract_state_value = pose_extract_state_values
    else: raise Exception(f"Pose size: {curr_poses.size(1)} is not 2 or 3")

    if sparse:
        _values = extract_state_value(action_values, curr_poses)
        return F.cross_entropy(_values, curr_actions)
    else:
        _values = extract_state_value(action_values, poses, index)
        return (F.cross_entropy(_values, actions, reduction="none") * discounts).sum() / batch_sz


def pos_motion_loss(motion_logit, *, _poses=None, _next_poses=None, _actions=None, **kwargs):
    d_poses = _next_poses - _poses
    _, _, Kx, Ky = motion_logit.size()

    dxs, dys = d_poses[:, 0], d_poses[:, 1]
    kxs, kys = dxs + (Kx - 1) // 2, dys + (Ky - 1) // 2

    motion_mask = (kxs >= 0) & (kxs < Kx) & (kys >= 0) & (kys < Ky)
    motion_labels = kxs * Ky + kys  # (batch,) [0...kx*ky)
    motion_labels = motion_labels[motion_mask].long()

    _p_logits = motion_logit[_actions][motion_mask]
    _p_logits = rearrange(_p_logits, "b () kx ky -> b (kx ky)")

    return F.cross_entropy(_p_logits, motion_labels)


def pose_motion_loss(motion_logit, *, _poses=None, _next_poses=None, _actions=None, **kwargs):
    d_poses = _next_poses - _poses
    _, _, _, Kx, Ky = motion_logit.size()

    _ori_curr = _poses[:, 0]
    _ori_next = _next_poses[:, 0]
    dxs, dys = d_poses[:, 1], d_poses[:, 2]
    kxs, kys = dxs + (Kx - 1) // 2, dys + (Ky - 1) // 2

    motion_mask = (kxs >= 0) & (kxs < Kx) & (kys >= 0) & (kys < Ky)
    motion_labels = _ori_next * Kx * Ky + kxs * Ky + kys  # (batch,) [0...ori*kx*ky)
    motion_labels = motion_labels[motion_mask].long()

    _p_logits = motion_logit[_actions, _ori_curr.long()][motion_mask]
    _p_logits = rearrange(_p_logits, "b o2 kx ky -> b (o2 kx ky)")

    return F.cross_entropy(_p_logits, motion_labels)


def make_conv_layers(l_i=None, l_h=None, l_o=None, kx=None, ky=None, n_layers=2, dropout=None):
    kernel_c_x = (kx - 1) // 2
    kernel_c_y = (ky - 1) // 2
    padding = (kernel_c_x, kernel_c_y)
    assert n_layers >= 2
    layers = [
            nn.Conv2d(in_channels=l_i, out_channels=l_h,
                      kernel_size=(kx, ky), stride=(1, 1), padding=padding, bias=True),
            nn.Dropout(dropout),
            nn.ReLU()
    ]
    for i in range(n_layers - 2):
        layers += [
            nn.Conv2d(in_channels=l_h, out_channels=l_h,
                      kernel_size=(kx, ky), stride=(1, 1), padding=padding, bias=True),
            nn.Dropout(dropout),
            nn.ReLU()
        ]
    layers += [
        nn.Conv2d(in_channels=l_h, out_channels=l_o,
                  kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
    ]
    return nn.Sequential(*layers)

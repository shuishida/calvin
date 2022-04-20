import torch
import torch.nn.functional as F
from einops import rearrange

from core.domains.gridworld.actions import GridDirs
from core.mdp.actions import ActionSetBase, EgoActionSetBase


def get_aa_manual(input_view, action_set: ActionSetBase):
    """
    :param input_view: (batch_sz, l_i, map_x, map_y)
    :return: available actions A(s, a): (batch_sz, n_actions, map_x, map_y)
    """
    aaw = torch.zeros((9, 3, 3, 3)).to(input_view.device)
    aab = torch.zeros((9,)).to(input_view.device)
    for i, (a_x, a_y) in enumerate(action_set):
        aaw[i, 0, 1, 1] = -10.0
        if a_x == 0 and a_y == 0:
            aaw[i, 1, 1, 1] = -0.5
            aaw[i, 2, 1, 1] = 1.0
        elif a_x == 0 or a_y == 0:
            aaw[i, 1, 1, 1] = 1.0
            aaw[i, 0, a_x + 1, a_y + 1] = -1.5
        else:
            aaw[i, 1, 1, 1] = 1.5
            aaw[i, 0, a_x + 1, a_y + 1] = -2.0
            aaw[i, 0, 1, a_y + 1] = -1.0
            aaw[i, 0, a_x + 1, 1] = -1.0
    aa = F.conv2d(input_view, aaw, bias=aab, stride=1, padding=1)
    return torch.where(aa >= 0, torch.ones_like(aa), torch.zeros_like(aa))


def get_aa_ego_manual(input_view, dirs: GridDirs,
                      action_set: EgoActionSetBase,
                      kernel_size=3):
    """
    :param input_view: (batch_sz, l_i, map_x, map_y)
    :return: available actions A(s, a): (batch_sz, n_actions, ori, map_x, map_y)
    """
    n_actions = len(action_set)    # 5
    aaw = torch.zeros((n_actions, len(dirs), 3, kernel_size, kernel_size)).to(input_view.device)
    aab = torch.zeros((n_actions, len(dirs))).to(input_view.device)
    kernel_centre = kernel_size // 2
    aaw[:, :, 0, kernel_centre, kernel_centre] = -10.0           # if you are already on an obstacle you're doomed
    for i, action in enumerate(action_set):
        if action not in [action_set.move_forward, action_set.turn_right, action_set.turn_left]:
            aab[i, :] = -100
            continue
        for j, (dir_x, dir_y) in enumerate(dirs):
            for step_size in range(1, kernel_centre + 1):
                new_dir, new_x, new_y = get_new_state(((dir_x, dir_y), kernel_centre, kernel_centre), action,
                                                      dirs=dirs,
                                                      action_set=action_set,
                                                      step_size=step_size)
                aaw[i, j, 0, new_x, new_y] = -10.0
                aaw[i, j, 2, new_x, new_y] = 1.5
                aaw[i, j, 1, new_x, new_y] = 1.5
                if kernel_centre != new_x and kernel_centre != new_y:
                    aaw[i, j, 0, kernel_centre, new_y] = -1.0
                    aaw[i, j, 0, new_x, kernel_centre] = -1.0
    aaw = rearrange(aaw, "n o i kx ky -> (n o) i kx ky")
    aa = F.conv2d(input_view, aaw, bias=aab.view(-1), stride=1, padding=kernel_centre)
    aa = torch.where(aa >= 0, torch.ones_like(aa), torch.zeros_like(aa))
    return rearrange(aa, "b (a o) x y -> b a o x y", o=len(dirs))


def get_aa_from_obst_map(obstacle_map, action_set: ActionSetBase):
    """
    :param obstacle_map: (batch_sz, 1, map_x, map_y)
    :return: available actions A(s, a): (batch_sz, n_actions, map_x, map_y)
    """
    batch_sz, _, map_x, map_y = obstacle_map.size()
    o_padded = F.pad(obstacle_map, [1, 1, 1, 1])
    aa_stack = []
    for action in action_set:
        if action == action_set.done:
            aa = torch.zeros_like(obstacle_map)
        else:
            a_x, a_y = action
            i, j = a_x + 1, a_y + 1
            o_next = o_padded[..., i:i + map_x, j:j + map_y]
            aa = (1.0 - obstacle_map) * (1.0 - o_next)
            if a_x != 0 and a_y != 0:
                o_x_next = o_padded[..., i:i + map_x, 1:1 + map_y]
                o_y_next = o_padded[..., 1:1 + map_x, j:j + map_y]
                aa *= (1.0 - o_x_next * o_y_next)
        aa_stack.append(aa)
    return torch.cat(aa_stack, dim=1)


def get_obst_map_manual(input_view, conf=1.0):
    obstacles = input_view[:, 0]
    obstacles = obstacles * conf + (1 - obstacles) * (1 - conf)
    return obstacles.unsqueeze(1)


def get_unobserved_map_manual(input_view, conf=1.0):
    obstacles, free = input_view[:, 0], input_view[:, 1]
    unknown = 1 - obstacles - free
    unknown = unknown * conf + (1 - unknown) * (1 - conf)
    return unknown.unsqueeze(1)

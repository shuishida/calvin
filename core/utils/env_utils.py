from enum import Enum
from typing import Tuple, Any

import numpy as np

from core.mdp.actions import EgoActionSetBase


def check_repeat(path, offset=5):
    rev_path = list(reversed(path))
    curr_state = rev_path[0]
    if len(rev_path) > 2 * offset:
        if curr_state in rev_path[offset:]:
            repeat_index = rev_path[offset:].index(curr_state) + offset
            if rev_path[:repeat_index] == rev_path[repeat_index:2 * repeat_index]:
                return rev_path[:repeat_index]
    return None


def rotate_align_agent(agent_angle, goal_angle, turn_angle):
    angle_diff = (goal_angle - agent_angle) % (2 * np.pi)
    if angle_diff < np.pi:
        n_turns = int(angle_diff / turn_angle + 0.5)
    else:
        n_turns = - int((2 * np.pi - angle_diff) / turn_angle + 0.5)
    return n_turns


def update_env(meta, env, action, *, curr_dir=None, turn_step=None, action_set: EgoActionSetBase = None) -> Tuple[Any, float, bool, Any]:
    """
    :param meta:
    :param env: MiniWorldEnv    Note: the env will be mutated upon update
    :param state:
    :param action:
    :return: (obsv, reward, done, info)
    """
    given_action = action
    # if meta.include_ori:
    #     _, reward, done, _ = env.step(given_action)
    #     return reward, done
    if given_action == meta.actions.done:
        return env.step(action=action_set.done)

    total_reward = 0
    n_turns = rotate_align_agent(curr_dir, meta.actions.dir(given_action), turn_step)
    action = action_set.turn_left if n_turns > 0 else action_set.turn_right
    for _ in range(int(abs(n_turns))):
        obsv, reward, done, info = env.step(action=action)
        total_reward += reward
        if done: return obsv, total_reward, True, info
    obsv, reward, done, info = env.step(action=action_set.move_forward)
    total_reward += reward
    return obsv, total_reward, done, info


class NavStatus(Enum):
    success = 0
    in_progress = 1
    invalid_action = 2
    repeat = 3
    false_complete = 4
    max_step_exceeded = 5

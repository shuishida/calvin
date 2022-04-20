from typing import Tuple, Dict, Any, Optional

import numpy as np
import random
from collections import defaultdict

from core.utils.planner_utils import min_cost_states
from core.utils.tensor_utils import random_choices


class Planner:
    def __init__(self, meta, states):
        self.meta = meta
        self.states = states

    def sample_states(self, n_samples, replace=False):
        return random_choices(self.states, n_samples, replace=replace)

    def transition(self, curr_state, action, reverse=False):
        """
        :param curr_state:
        :param action:
        :param reverse: if reversed, give previous state instead of next state
        :return: list of (next_state, action, cost) tuples, (prev_state, action, cost) if reversed
        """
        raise NotImplementedError

    def transitions(self, curr_state, reverse=False):
        """
        :param curr_state:
        :param reverse: if True, return prev states rather than next states
        :return: list of (next_state, action, cost) tuples, (prev_state, action, cost) if reversed
        """
        transitions = []
        for action in self.meta.actions:
            trans = self.transition(curr_state, action, reverse=reverse)
            if trans is not None: transitions.append(trans)
        return transitions

    def get_trajectory(self, source, target):
        """
        :param source: source state
        :param target: target state
        :return: list of tuples (state, action) along trajectory. last pair is (target, done)
        """
        cost, best_trans, target = self.get_transition_tree(source, is_root_target=False, terminal=target)
        if cost[target] == np.inf: return []
        state = target
        path = [target]
        actions = [self.meta.actions.done]
        while state != source:
            state, action = best_trans[state]
            path.append(state)
            actions.append(action)
        path = list(reversed(path))
        actions = list(reversed(actions))
        return list(zip(path, actions, path[1:] + [path[-1]]))

    def sample_trajectories(self, min_traj_len, n_trajs=1, target=None):
        """
        :param min_traj_len: minimum threshold for trajectory length
        :param n_trajs: number of trajectories to be sampled
        :param target: target state
        :return: list of trajectories, where each trajectory is a list of tuples (state, action) along trajectory
        """
        trajs = []
        for _ in range(n_trajs * 8):
            if len(trajs) == n_trajs:
                break
            if target is None:
                source, target = self.sample_states(n_samples=2)
            else:
                source, = self.sample_states(n_samples=1)
            traj = self.get_trajectory(source, target)
            if len(traj) >= min_traj_len + 1:
                trajs.append(traj)
        return trajs

    def terminal_condition(self, state, target):
        return state == target

    def get_priority(self, cost, state, target):
        return cost

    def get_state_cost(self, state):
        return 0

    def get_transition_tree(self, root, is_root_target=True, terminal=None) -> \
            Tuple[Dict[Any, float], Dict[Any, Optional[Tuple[Any, Any]]], Any]:
        """
        Dijkstra's algorithm
        :param root:
        :param is_root_target:
        :return:
        """
        best_transition: Dict[Any, Optional[Tuple[Any, Any]]] = defaultdict(lambda: None)
        cost_so_far = defaultdict(lambda: np.inf)
        frontier = {root: 0}
        cost_so_far[root] = 0
        curr_state = root

        while frontier:
            # node with the least cost selected first
            min_states, min_cost = min_cost_states(frontier)
            if min_cost == np.inf:
                # all remaining states are unreachable
                break
            curr_state = random.choice(min_states)
            priority = frontier.pop(curr_state)
            if self.terminal_condition(curr_state, terminal):
                break
            for new_state, action, trans_cost in self.transitions(curr_state, reverse=is_root_target):
                new_state_new_cost = cost_so_far[curr_state] + trans_cost + self.get_state_cost(new_state)
                new_state_curr_cost = cost_so_far[new_state]
                if new_state_new_cost < new_state_curr_cost:
                    # a path with smaller cost to next_state has been found
                    cost_so_far[new_state] = new_state_new_cost
                    frontier[new_state] = self.get_priority(new_state_new_cost, curr_state, terminal)
                    best_transition[new_state] = (curr_state, action)

        return cost_so_far, best_transition, curr_state

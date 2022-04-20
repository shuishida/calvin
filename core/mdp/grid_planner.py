import numpy as np
from core.mdp.planner import Planner


class GridPlanner(Planner):
    def __init__(self, meta, states, grid):
        self.grid = grid
        super(GridPlanner, self).__init__(meta, states)

    def get_motion(self, state_index, next_state_index, i):
        return next_state_index[i] - state_index[i]

    def get_values_and_best_actions(self, target):
        meta = self.meta
        cost, best_trans, _ = self.get_transition_tree(target, is_root_target=True)
        values = np.ones(meta.state_shape, dtype=np.float) * -np.inf
        counts = np.zeros(meta.state_shape, dtype=np.int)
        best_actions = np.zeros((len(meta.actions), *meta.state_shape), dtype=bool)
        motion = np.zeros((len(meta.actions), len(meta.state_shape) + 1, *meta.state_shape), dtype=np.int)
        for state in self.states:
            state_index = meta.state_to_index(state)
            values[state_index] = - cost[state]
            if best_trans[state]:
                _, action = best_trans[state]
                best_actions[(meta.action_to_index(action), *state_index)] = True
                while best_trans[state]:
                    next_state, action = best_trans[state]
                    state_index = meta.state_to_index(state)
                    next_state_index = meta.state_to_index(next_state)
                    counts[state_index] += 1
                    for i in range(len(meta.state_shape)):
                        motion[(meta.action_to_index(action), i, *state_index)] \
                            = self.get_motion(state_index, next_state_index, i)
                    motion[(meta.action_to_index(action), len(meta.state_shape), *state_index)] = 1
                    state = next_state
            state_index = meta.state_to_index(state)
            counts[state_index] += 1
        target_index = meta.state_to_index(target)
        done_index = meta.action_to_index(meta.actions.done)
        best_actions[(done_index, *target_index)] = True
        for i in range(len(meta.state_shape)):
            motion[(done_index, i, *target_index)] = self.get_motion(target_index, target_index, i)
        motion[(done_index, len(meta.state_shape), *target_index)] = 1
        return values, best_actions, counts, motion

    def get_valid_actions(self, target):
        meta = self.meta
        vaild_actions = np.zeros((len(meta.actions), *meta.state_shape), dtype=bool)
        for state in self.states:
            state_index = meta.state_to_index(state)
            for _, action, _ in self.transitions(state):
                vaild_actions[(meta.action_to_index(action), *state_index)] = True
        target_index = meta.state_to_index(target)
        done_index = meta.action_to_index(meta.actions.done)
        vaild_actions[(done_index, *target_index)] = True
        return vaild_actions

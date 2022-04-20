import numpy as np


def min_cost_states(cost: dict):
    min_states = []
    min_cost = np.inf
    for state, state_cost in cost.items():
        if state_cost < min_cost:
            min_states = [state]
            min_cost = state_cost
        elif state_cost == min_cost:
            min_states.append(state)
    return min_states, min_cost


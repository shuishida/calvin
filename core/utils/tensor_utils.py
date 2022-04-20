import random

import numpy as np
import torch


def to_numpy(data: dict):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cpu().data.numpy()
    return data


def random_choice(array):
    return array[np.random.randint(len(array))]


def random_choices(array, n_samples, replace=False):
    if replace:
        return [random.choice(array) for _ in range(n_samples)]
    else:
        assert n_samples <= len(array), \
            "number of samples required exceeds the number of states. consider replace == True"
        indices = np.arange(len(array))
        np.random.shuffle(indices)
        return [array[i] for i in indices[:n_samples]]

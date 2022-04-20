from collections import defaultdict

import torch
import random
import numpy as np
from gym.utils.seeding import create_seed


def set_random_seed(seed: int = None):
    if seed is None:
        seed = create_seed()
    else:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)


def count_tensors_shape(objs):
    count_shapes = defaultdict(int)
    for obj in objs:
        count_shapes[tuple(obj.size())] += 1
    return count_shapes


def print_items(*row, width=12):
    print(format_table_row(row, width))


def format_table_row(row, width):
    out = " | ".join(format_item(x, width) for x in row)
    return out


def format_item(x, width):
    if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        x = x.item()
    if isinstance(x, float): rep = f"{x:.{width-5}}"
    elif x is None: rep = "N/A"
    else: rep = str(x)
    return " " * (width - len(rep)) + rep


class Stats:
    def __init__(self):
        self._data = defaultdict(list)

    def add(self, key, value):
        self._data[key].append(value)

    def __getitem__(self, item):
        dat = self._data[item]
        return dat if dat else None

    def add_all(self, d):
        for k, v in d.items():
            self.add(k, v)

    def mean(self, key):
        return np.mean(self._data[key])

    def sum(self, key):
        return np.sum(self._data[key])

    def means(self):
        return {k: self.mean(k) for k, v in self._data.items()}

    def sums(self):
        return {k: self.sum(k) for k, v in self._data.items()}

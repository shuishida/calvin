import numpy as np
import argparse

from core.utils.plot_utils import visualise


class GridMap:
    def __init__(self, state_shape: tuple):
        self.state_shape = state_shape            # (size_x, size_y)
        self.grid = np.zeros(self.state_shape)   # 0 if empty, 1 if full

    def fill(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def show(self, opt_path=None, pred_path=None, axes=None, titles=None):
        visualise([self.grid], opt_path=opt_path, pred_path=pred_path, axes=axes, titles=titles)

    def __repr__(self):
        x, y = self.state_shape
        return f"{self.__class__.__name__}_{x}x{y}"

    def __str__(self):
        header = "   " + "".join([str(j % 10) for j in range(self.grid.shape[1])])
        content = "\n".join([f"{i:02d} " + "".join(["X" if e else " " for j, e in enumerate(line)]) for i, line in enumerate(self.grid)])
        return f"{self.__class__.__name__} [size: {self.state_shape}]:\n{header}\n{content}\n{header}"


def add_gridmap_args(p: argparse.ArgumentParser):
    p.add_argument("--size", "-sz", help="map size", type=int, nargs=2, default=(30, 30))
    return p


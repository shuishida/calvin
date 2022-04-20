"""
Code based on Kent Sommer's implementation of Value Iteration Networks
https://github.com/kentsommer/pytorch-value-iteration-networks
Adapted by Shu Ishida (https://github.com/shuishida)
Circular obstacle features newly added
"""

import numpy as np
import argparse

import sys

sys.path.append('.')

from core.domains.gridworld.map.obstacle_map.obstacle_utils import gen_random_circle, gen_random_rect
from core.domains.gridworld.map.gridmap import GridMap, add_gridmap_args


class ObstacleMap(GridMap):
    """A class for generating obstacles in a domain"""

    def __init__(self,
                 state_shape: tuple,
                 mask: list = None,
                 max_num_obst: int = 50,
                 max_obst_size: int = None,
                 obst_types=None):
        super(ObstacleMap, self).__init__(state_shape)
        # mask: cells that should not be filled by obstacles. (x, y) or [[list of x coords], [list of y coords]]
        self.mask = mask
        if isinstance(obst_types, str):
            obst_types = [obst_types]
        self.obst_types = obst_types or ["circ", "rect"]
        self.max_num_obst = max_num_obst
        self.max_obst_size = max_obst_size or max(self.state_shape) // 4

    def fill(self, refresh_mask=True):
        if refresh_mask:
            x_max, y_max = self.state_shape
            # select a goal randomly that doesn't overlap with the border
            self.mask = np.random.randint((1, 1), (x_max - 1, y_max - 1))
        self.init_grid()
        self.add_n_rand_obs()
        return self.grid

    def init_grid(self, grid=None):
        if grid is None: grid = np.zeros(self.state_shape)
        # make full outer border an obstacle
        grid[:, [0, -1]] = 1
        grid[[0, -1], :] = 1
        if not self.check_mask(grid):
            raise Exception("Boarder cannot be added because mask interferes")
        self.grid = grid

    def check_mask(self, grid=None):
        if self.mask is None: return True
        if grid is None: grid = self.grid
        # e goal is in free space
        return not np.any(grid[self.mask[0], self.mask[1]])

    def add_rand_obst(self, obj_type):
        # add random (valid) obstacle to map
        if obj_type == "circ":
            obst = gen_random_circle(self.state_shape, self.max_obst_size // 2)
        elif obj_type == "rect":
            obst = gen_random_rect(self.state_shape, self.max_obst_size)
        else:
            raise Exception(f"obstacle type {obj_type} not recognised")
        im_try = np.logical_or(self.grid, obst)
        success = self.check_mask(im_try)
        if success:
            self.grid = im_try
        return success

    def add_n_rand_obs(self, n=None):
        # add random (valid) obstacles to map
        for _ in range(n or self.max_num_obst):
            obj_type = np.random.choice(self.obst_types)
            self.add_rand_obst(obj_type)


def add_obstacle_map_args(p: argparse.ArgumentParser):
    add_gridmap_args(p)
    p.add_argument("--max_num_obst", "-n_obs", help="maximum number of obstacles to be placed", default=50, type=int)
    p.add_argument("--max_obst_size", "-obs_sz", help="maximum size of obstacles", default=None, type=int),
    p.add_argument("--obst_types", help="list of types of obstacle ('circ', 'rect', ['circ', 'rect'])",
                   nargs='+', default=None, type=str)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_obstacle_map_args(parser)
    args = parser.parse_args()

    gridmap = ObstacleMap(args.size, max_num_obst=args.max_num_obst, max_obst_size=args.max_obst_size, obst_types=args.obst_types)
    for i in range(10):
        gridmap.fill()
        gridmap.show()

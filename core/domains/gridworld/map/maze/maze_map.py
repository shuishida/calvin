import numpy as np
import argparse
import sys

sys.path.append(".")

from core.domains.gridworld.map.gridmap import GridMap, add_gridmap_args
from core.domains.gridworld.map.maze.maze_gen import MazeGenerator


class MazeMap(GridMap):
    def __init__(self, state_shape: tuple, path_thickness: int = 1):
        super(MazeMap, self).__init__(state_shape)
        self.path_thickness = path_thickness
        self.n_rows, self.n_cols = (np.array(state_shape) - 1) // (2 * path_thickness)

    def fill(self):
        maze = MazeGenerator(self.n_rows, self.n_cols, self.path_thickness).grid()
        maze_x_max, maze_y_max = maze.shape
        grid_x_max, grid_y_max = self.grid.shape
        self.grid = np.ones(self.state_shape)
        self.grid[:maze_x_max, :maze_y_max] = maze[:grid_x_max, :grid_y_max]
        return self.grid


def add_mazemap_args(p: argparse.ArgumentParser):
    add_gridmap_args(p)
    p.add_argument("--p_thick", "-th", help="path thickness", default=1, type=int)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_mazemap_args(parser)
    args = parser.parse_args()

    gridmap = MazeMap(args.size, path_thickness=args.thick)
    for i in range(10):
        gridmap.fill()
        gridmap.show()

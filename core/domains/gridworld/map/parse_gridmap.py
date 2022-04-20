import sys
import argparse

sys.path.append('.')

from core.domains.gridworld.map.floormap.floormap import add_floormap_args, FloorMap
from core.domains.gridworld.map.maze.maze_map import add_mazemap_args, MazeMap
from core.domains.gridworld.map.obstacle_map.obstacle_map import add_obstacle_map_args, ObstacleMap


def add_map_args(parser: argparse.ArgumentParser):
    argv = sys.argv
    assert '--map' in argv, "--map should be passed as an argument"
    map_type = argv[argv.index('--map') + 1]

    map_parser = parser.add_argument_group("grid map parser")
    map_parser.add_argument("--map", help="map type ('maze', 'obst', 'floor')", type=str, default='obst')

    if map_type == 'maze':
        add_mazemap_args(map_parser)
    elif map_type == 'obst':
        add_obstacle_map_args(map_parser)
    elif map_type == 'floor':
        add_floormap_args(map_parser)


def get_map(map_type, *, size=None, max_num_obst=None, max_obst_size=None, obst_types=None,
            min_room_size=None, p_thick=1, w_thick=1, **kwargs):
    if map_type == 'maze':
        gridmap = MazeMap(size, path_thickness=p_thick)
    elif map_type == 'obst':
        gridmap = ObstacleMap(size, max_num_obst=max_num_obst, max_obst_size=max_obst_size, obst_types=obst_types)
    elif map_type == 'floor':
        gridmap = FloorMap(size, min_room_size=min_room_size, path_thickness=p_thick, wall_thickness=w_thick)
    else:
        raise Exception("no matching map type")
    return gridmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_map_args(parser)
    args = parser.parse_args()

    gridmap = get_map(args.map, **vars(args))

    for i in range(10):
        gridmap.fill()
        gridmap.show()

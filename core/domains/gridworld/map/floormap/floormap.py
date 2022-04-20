import numpy as np
import argparse
import sys

sys.path.append(".")

from core.domains.gridworld.map.floormap.floormap_utils import RectNode, Room, Walls
from core.domains.gridworld.map.gridmap import GridMap, add_gridmap_args


class FloorMap(GridMap):
    def __init__(self, state_shape: tuple, min_room_size: int,
                 path_thickness: int = 1, wall_thickness: int = 1):
        super(FloorMap, self).__init__(state_shape)
        self.p_thick = path_thickness
        self.w_thick = wall_thickness
        self.min_room_size = min_room_size
        self.rooms = []

    def init_room(self) -> RectNode:
        self.grid = np.ones(self.state_shape)
        self.rooms = []
        size_x, size_y = self.state_shape
        th = self.p_thick
        return RectNode(th, th, size_x - th, size_y - th)

    def create_rooms(self, root: RectNode):
        children = root.split(self.p_thick, min_room_size=self.min_room_size)
        if children is None:
            room = Room.from_rect_node(root, id=len(self.rooms) + 1)
            self.rooms.append(room)
        else:
            for child in children:
                self.create_rooms(child)

    def get_room_map(self):
        room_map = np.zeros(self.state_shape, dtype=object)
        for room in self.rooms:
            x1, y1, x2, y2 = room.bbox
            room_map[x1:x2, y1:y2] = room
        return room_map

    def fill(self):
        root = self.init_room()
        self.create_rooms(root)
        room_map = self.get_room_map()
        walls = Walls(room_map, self.w_thick)
        walls.add_doors_from_rectnode(root, self.p_thick)
        self.grid = room_map == 0
        for wall in walls.values():
            if wall.door is not None:
                x1, y1, x2, y2 = wall.door.bbox
                self.grid[x1:x2, y1:y2] = 0

        return self.grid


def add_floormap_args(p: argparse.ArgumentParser):
    add_gridmap_args(p)
    p.add_argument("--min_room_size", "-minroom", help="minimum threshold for room size", default=2, type=int)
    p.add_argument("--p_thick", "-pth", help="path thickness", default=1, type=int)
    p.add_argument("--w_thick", "-wth", help="wall thickness", default=1, type=int)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_floormap_args(parser)
    args = parser.parse_args()

    gridmap = FloorMap(args.size, min_room_size=args.min_room_size,
                       path_thickness=args.p_thick, wall_thickness=args.w_thick)
    for i in range(10):
        gridmap.fill()
        gridmap.show()

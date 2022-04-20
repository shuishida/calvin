import random
import numpy as np


def intersect_dim(a1_s, a2_s, a1_o, a2_o):
    if a1_o <= a1_s < a2_o:
        return a1_s, a2_o
    elif a1_s <= a1_o < a2_s:
        return a1_o, a2_s
    else:
        return None


def contains_dim(a1_s, a2_s, a1_o, a2_o):
    return a1_s <= a1_o <= a2_o <= a2_s


def order_dim(a1_s, a2_s, a1_o, a2_o):
    if intersect_dim(a1_s, a2_s, a1_o, a2_o): return None
    if a2_s < a1_o:
        return a1_s, a2_s, a1_o, a2_o
    elif a2_o < a1_s:
        return a1_o, a2_o, a1_s, a2_s
    else:
        return None


class Rect:
    def __init__(self, x1, y1, x2, y2):
        assert x1 < x2, "x1 must be smaller than x2"
        assert y1 < y2, "y1 must be smaller than y2"
        self.bbox = (x1, y1, x2, y2)

    def _assemble_dim(self, other, dim):
        x1_s, y1_s, x2_s, y2_s = self.bbox
        x1_o, y1_o, x2_o, y2_o = other.bbox
        if dim == "x":
            return x1_s, x2_s, x1_o, x2_o
        elif dim == "y":
            return y1_s, y2_s, y1_o, y2_o
        raise Exception("dim must be either x or y")

    def order_dim(self, other, dim):
        return order_dim(*self._assemble_dim(other, dim))

    def intersect_dim(self, other, dim):
        return intersect_dim(*self._assemble_dim(other, dim))

    def intersect(self, other):
        return self.intersect_dim(other, "x") and self.intersect_dim(other, "y")

    def contains_dim(self, other, dim):
        return contains_dim(*self._assemble_dim(other, dim))

    def contains(self, other):
        return self.contains_dim(other, "x") and self.contains_dim(other, "y")


class RectNode(Rect):
    def __init__(self, x1, y1, x2, y2, parent=None):
        super(RectNode, self).__init__(x1, y1, x2, y2)
        self.parent: RectNode or None = parent
        self.children = None
        self.direction = None
        self.depth = parent.depth + 1 if parent else 0
        self.wall: Rect or None = None

    def split(self, thick: int = 1, min_room_size: int = 1):
        """
        :param thick: wall thickness
        :return:
        """
        x1, y1, x2, y2 = self.bbox
        x_margin = x2 - x1 - 2 * min_room_size - thick
        y_margin = y2 - y1 - 2 * min_room_size - thick
        if x_margin < 0 and y_margin < 0: return None
        elif x_margin == y_margin: direction = random.choice(['h', 'v'])
        else: direction = 'h' if x_margin > y_margin else 'v'
        l1, l2 = (x1, x2) if direction == 'h' else (y1, y2)
        wall_start_min = l1 + min_room_size
        wall_start_max = l2 - min_room_size - thick
        wall_start = random.randint(wall_start_min, wall_start_max)
        wall_end = wall_start + thick
        if direction == 'h':
            child1 = RectNode(x1, y1, wall_start, y2, self)
            child2 = RectNode(wall_end, y1, x2, y2, self)
            self.wall = Rect(wall_start, y1, wall_end, y2)
        else:
            child1 = RectNode(x1, y1, x2, wall_start, self)
            child2 = RectNode(x1, wall_end, x2, y2, self)
            self.wall = Rect(x1, wall_start, x2, wall_end)
        self.direction = direction
        self.children = (child1, child2)
        return self.children


class Room(Rect):
    def __init__(self, x1, y1, x2, y2, id):
        super(Room, self).__init__(x1, y1, x2, y2)
        self.id = id

    def __hash__(self):
        return self.id, self.bbox

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__hash__() == other.__hash__()

    @classmethod
    def from_rect_node(cls, node: RectNode, id: int):
        return Room(*node.bbox, id=id)


class Wall(Rect):
    def __init__(self, room1: Room, room2: Room):
        if room1.id > room2.id:
            room1, room2 = room2, room1
        self.rooms = (room1, room2)
        self.door: Rect or None = None

        for length_dim, thick_dim in [("x", "y"), ("y", "x")]:
            l_coords = room1.intersect_dim(room2, length_dim)
            if l_coords:
                l1, l2 = l_coords
                _, t1, t2, _ = room1.order_dim(room2, thick_dim)
                self.length = l2 - l1
                self.thick = t2 - t1
                self.dim = length_dim
                break
        else: raise Exception("No intersection found between room 1 and room 2")

        if self.dim == "x":
            bbox = (l1, t1, l2, t2)     # (x1, y1, x2, y2)
        elif self.dim == "y":
            bbox = (t1, l1, t2, l2)     # (x1,( y1, x2, y2)
        else: raise Exception("self.dim does not match 'x' or 'y'")

        super(Wall, self).__init__(*bbox)

    def __str__(self):
        return f"Wall {self.rooms[0].id, self.rooms[1].id}"

    def add_door(self, door_width):
        assert self.door is None, "door already exists"
        offset = random.randint(0, self.length - door_width)
        x1, y1, x2, y2 = self.bbox
        if self.dim == "x":
            self.door = Rect(x1 + offset, y1, x1 + offset + door_width, y2)
        elif self.dim == "y":
            self.door = Rect(x1, y1 + offset, x2, y1 + offset + door_width)
        else:
            raise Exception("self.dim does not match 'x' or 'y'")


class Walls(dict):
    def __init__(self, room_map, w_thick):
        """
        :param room_map: numpy 2d map where cells point to Room objects
        :param w_thick:
        """
        super(Walls, self).__init__()
        self.wall_map = np.zeros_like(room_map, dtype=object)

        size_x, size_y = room_map.shape
        for i in range(w_thick, size_x - w_thick):
            for j in range(w_thick, size_y - w_thick):
                if not room_map[i, j]:
                    for (i1, j1), (i2, j2), dim_thick in [
                        [(i - w_thick, j), (i + w_thick, j), "x"],
                        [(i, j - w_thick), (i, j + w_thick), "y"]]:
                        room1 = room_map[i1, j1]
                        room2 = room_map[i2, j2]
                        if room1 and room2:
                            _, a1, a2, _ = room1.order_dim(room2, dim_thick)
                            if a2 - a1 == w_thick:
                                wall = self.find(room1, room2)
                                if not wall:
                                    wall = self.add(room1, room2)
                                self.wall_map[i, j] = wall

    def add(self, room1, room2):
        assert room1.id != room2.id, "room1 and room2 have the same id"
        if room1.id > room2.id:
            room1, room2 = room2, room1
        wall = self.find(room1, room2)
        if wall: raise Exception(f"{wall} already exists")
        wall = Wall(room1, room2)
        self[(room1.id, room2.id)] = wall
        return wall

    def find(self, room1: Room, room2: Room):
        id1, id2 = room1.id, room2.id
        if id1 > id2: id1, id2 = id2, id1
        return self.get((id1, id2))

    def add_doors_from_rectnode(self, node: RectNode, door_width):
        if not node.wall: return
        x1, y1, x2, y2 = node.wall.bbox
        walls = self.wall_map[x1:x2, y1:y2]
        walls = walls[walls != 0]
        if len(walls):
            wall = random.choice(walls)
            wall.add_door(door_width)
        if node.children is not None:
            for child in node.children:
                self.add_doors_from_rectnode(child, door_width)

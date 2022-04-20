from core.mdp.actions import ActionSetBase


class GridDirs(list):
    DIRS_8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    DIRS_4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self, four_way=False):
        dirs = self.DIRS_4 if four_way else self.DIRS_8
        self.rotate_clockwise = {}
        self.rotate_anticlockwise = {}
        for dir1, dir2 in zip(dirs, dirs[1:] + dirs[:1]):
            self.rotate_anticlockwise[dir1] = dir2
            self.rotate_clockwise[dir2] = dir1
        super(GridDirs, self).__init__(dirs)

    def rotate(self, direction, clockwise=None):
        assert clockwise is not None, "Rotation direction not specified."
        return self.rotate_clockwise[direction] if clockwise else self.rotate_anticlockwise[direction]

    def opposite(self, direction):
        l = len(self)
        return self[(self.index(direction) + l // 2) % l]

    def degrees(self, direction):
        if len(self) == 8:
            return self.index(direction) * 45
        else:
            return self.index(direction) * 90


class GridActionSet(ActionSetBase):
    def __init__(self, four_way=False):
        self.dirs = GridDirs(four_way)
        self._done = (0, 0)
        super(GridActionSet, self).__init__(list(self.dirs) + [self._done])

    @property
    def done(self):
        return self._done

import numpy as np
import random


class Cell(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.neighbours = []
        self.connected = []

    def __str__(self):
        return f"Cell {self.h, self.w}"

    def connect(self, other):
        assert other in self.neighbours, "node not neighbouring"
        assert other not in self.connected, "node already connected"
        self.connected.append(other)
        other.connected.append(self)


class MazeGenerator(object):
    def __init__(self, H, W, path_thickness: int):
        self.H = H
        self.W = W
        self.thickness = path_thickness
        self.cells = [[Cell(h, w) for w in range(W)] for h in range(H)]
        self.visited = np.zeros((H, W), dtype=bool)
        for h in range(H):
            for w in range(W):
                cell = self.cells[h][w]
                cell.neighbours = self.get_neighbours(cell)
        self.connect()

    def connect(self):
        self.visited[tuple(self.random_point())] = True
        while not self.visited.all():
            self.add_path()

    def is_valid(self, h, w):
        return 0 <= h < self.H and 0 <= w < self.W

    def get_neighbours(self, cell: Cell):
        h, w = cell.h, cell.w
        return [self.cells[_h][_w] for (_h, _w) in [(h-1, w), (h+1, w), (h, w-1), (h, w+1)] if self.is_valid(_h, _w)]

    def random_point(self):
        return np.random.randint((self.H, self.W))

    def add_path(self):
        unvisited = np.argwhere(self.visited == False)
        h, w = unvisited[np.random.randint(len(unvisited))]
        curr = self.cells[h][w]
        path = [curr]
        while not self.visited[curr.h, curr.w]:
            new_node = random.choice(curr.neighbours)
            if new_node in path:
                curr = random.choice(path)
            else:
                path.append(new_node)
                curr.connect(new_node)
                curr = new_node
        for node in path:
            self.visited[node.h, node.w] = True

    def grid(self):
        """
        returns maze grid at its current state. to get a completed maze, run this after running the connect method
        """
        data = np.ones((self.H * 2 + 1, self.W * 2 + 1), dtype=bool)
        for h in range(self.H):
            for w in range(self.W):
                data[h * 2 + 1, w * 2 + 1] = not self.visited[h, w]
                cell = self.cells[h][w]
                for connected in cell.connected:
                    data[cell.h + connected.h + 1, cell.w + connected.w + 1] = False

        square_ones = np.ones((self.thickness, self.thickness))
        return np.kron(data, square_ones)

    def __str__(self):
        return "\n".join(["".join(["X" if e else " " for e in line]) for line in self.grid()])


if __name__ == "__main__":
    maze = MazeGenerator(8, 8, 2)
    print(maze)

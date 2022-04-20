import numpy as np


def gen_random_rect(state_shape: tuple, max_obs_size: int or tuple):
    grid = np.zeros(state_shape)
    if isinstance(max_obs_size, int or float):
        max_obs_size = (max_obs_size, max_obs_size)
    x_size, y_size = np.random.randint(max_obs_size)
    center_x, center_y = np.random.randint(state_shape)
    x1 = center_x - x_size // 2
    x2 = x1 + x_size
    y1 = center_y - y_size // 2
    y2 = y1 + y_size
    grid[x1:x2, y1:y2] = 1
    return grid


def gen_random_circle(state_shape: tuple, max_radius: int):
    grid = np.zeros(state_shape)
    x_max, y_max = state_shape
    c_x, c_y = np.random.randint(state_shape)
    r = np.random.randint(max_radius)
    for x in range(max(0, c_x - r), min(c_x + r + 1, x_max)):
        for y in range(max(0, c_y - r), min(c_y + r + 1, y_max)):
            if np.linalg.norm((x - c_x, y - c_y)) <= r:
                grid[x, y] = 1
    return grid

from typing import Tuple

import numpy as np


def sample_free_space(world_coords, cam_pos, n_samples_per_pixel=1):
    world_coords = np.expand_dims(world_coords, axis=-2).repeat(n_samples_per_pixel, axis=-2)  # (H, W, k, 3)
    camera_pos = cam_pos.reshape((1, 1, 1, 3))  # (1, 1, 1, 3)
    coeff = np.random.random(world_coords.shape) * 0.99  # (k, H, W, 3)
    sampled_points = world_coords * coeff + camera_pos * (1 - coeff)  # (H, W, k, 3)
    return sampled_points


def get_world_coord(depth, R, t, hfov):
    H, W = depth.shape
    us, vs = np.meshgrid(np.linspace(-1, 1, W) * np.tan(hfov / 2.) * W / H, np.linspace(-1, 1, H) * np.tan(hfov / 2.))
    return get_world_coord_from_mesh(us, vs, depth, R, t)


def get_world_coord_from_mesh(mesh_us, mesh_vs, depth, R, t):
    H, W = depth.shape

    # Unproject
    uvs = np.stack((mesh_us * depth, mesh_vs * depth, depth))
    uvs = uvs.reshape(3, -1)

    xyz = R.T @ uvs + t.reshape(3, 1)

    return xyz.reshape((3, H, W)).transpose((1, 2, 0))


def rotation_from_camera_normal(normal: np.ndarray):
    vertical = np.array([0, 1, 0])
    waxis = normal.reshape(-1)          # w axis is the normal vector, stretching out from the camera
    uaxis = np.cross(waxis, vertical)   # u axis is aligned with the image width, stretching right
    vaxis = np.cross(waxis, uaxis)      # v axis is aligned with the image height, stretching downwards
    R = np.stack([uaxis, vaxis, waxis])
    return R                            # from world -> camera


def distort_radius(rs, k1, k2, k3, k4):
    phi = np.arctan(rs)
    rds = phi + k1 * phi ** 3 + k2 * phi ** 5 + k3 * phi ** 7 + k4 * phi ** 9
    return rds


def get_undistorted_radius_mapping(distort_coeff: Tuple[float, float, float, float], step=0.001):
    """"
    numerically evaluating the inverse function of a fisheye distortion
    """

    rs = np.arange(0, 2, step)
    rds = distort_radius(rs, *distort_coeff)
    mapping = np.zeros_like(rds)
    next_rd = step
    i = 1
    for prev_r, prev_rd, r, rd in zip(rs[:-1], rds[:-1], rs[1:], rds[1:]):
        if i == len(mapping):
            break
        while rd >= next_rd:
            mapping[i] = ((rd - next_rd) * prev_r + (next_rd - prev_rd) * r) / (rd - prev_rd)
            i += 1
            next_rd += step
            if i == len(mapping):
                break

    return np.vectorize(lambda rd: mapping[int(rd / step)])


def undistort_mesh(mesh_xs, mesh_ys, distort_coeff: Tuple[float, float, float, float]):
    undistort_radius = get_undistorted_radius_mapping(distort_coeff)
    rds = (mesh_xs ** 2 + mesh_ys ** 2) ** 0.5
    rs = undistort_radius(rds)
    undistort = rs / rds
    return mesh_xs * undistort, mesh_ys * undistort

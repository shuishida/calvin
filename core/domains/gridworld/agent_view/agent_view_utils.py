#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shu Ishida   University of Oxford
#
# Distributed under terms of the MIT license.

import numpy as np
from math import pi, tan, ceil, floor, atan


def rotate90(tensor, times):
    """Rotate tensor counter-clockwise by 90 degrees a number of times.
    Assumes spatial dimensions are the last ones."""
    dim_x, dim_y = tensor.ndim - 2, tensor.ndim - 1
    transpose = lambda arr: np.transpose(arr, (*np.arange(dim_x), dim_y, dim_x))
    n_rotation = int(times % 4)
    if n_rotation == 1:  # 90 deg
        return transpose(np.flip(tensor, dim_x))
    elif n_rotation == 2:  # 180 deg
        return np.flip(np.flip(tensor, dim_x), dim_y)
    elif n_rotation == 3:  # 270 deg
        return transpose(np.flip(tensor, dim_y))
    else:  # 0 deg, no change
        assert n_rotation == 0
        return tensor


def extract_view(grid, h, w, view_range, ang90=0):
    """Extract a local view from an environment at the given pose"""
    # get coordinates of window to extract
    hs = np.arange(h - view_range, h + view_range + 1, dtype=int)
    ws = np.arange(w - view_range, w + view_range + 1, dtype=int)

    # get coordinate 0 instead of going out of bounds
    h_env, w_env = grid.shape[-2:]
    invalid_hs, invalid_ws = ((hs < 0) | (hs >= h_env), (ws < 0) | (ws >= w_env))  # coords outside the env
    hs[invalid_hs] = 0  # some valid index
    ws[invalid_ws] = 0  # some valid index

    # extract view, and set to 0 observations that were out of bounds
    # not equivalent to view = env[..., y1:y2, x1:x2]
    view = grid[..., hs, :][..., :, ws]

    view[..., invalid_hs, :] = -1  # out of the map
    view[..., :, invalid_ws] = -1  # out of the map

    # rotate. note only 90 degrees rotations are allowed
    return rotate90(view, ang90)


def embed_view(patch, state_shape, h0, w0, ang90=0):
    """Embed a local view in a global grid at the given pose"""
    patch = rotate90(patch, ang90)

    assert len(state_shape) == 2
    image = np.zeros((*patch.shape[:-2], *state_shape), dtype=patch.dtype)
    h_env, w_env = state_shape
    h_patch, w_patch = patch.shape[-2:]
    image[..., max(0, h0):h_patch + h0, max(0, w0):w_patch + w0] \
        = patch[..., max(0, -h0):h_env - h0, max(0, -w0):w_env - w0]

    return image


def get_partially_observable_pixels(square_patch, angle_ranges, split_required=True):
    """
    :param square_patch: 0-1 array of shape (2k+1, 2k+1), where obstacle-filled pixels are 1s
    :param angle_ranges:
        if split_required == True:
            list of tuples of feasible ranges of view
            [(a1, b1), (a2, b2), ...] where (a1 < b1), (a2 < b2), ...
        else:
            list of tuples of feasible ranges of view [(a1, b1, s1), (a2, b2, s2),...]
            where a < b and (a, b) fits into (-pi/4, pi/4), (pi/4, 3pi/4), (3pi/4, 5pi/4), (5pi/4, 7pi/4)
            and s denotes the side (0: right, 1: up, 2: left, 3: bottom)
    :param split_required: set to true for initial call to convert angle_ranges into the right format
        for subsequent recurrent calls:
    :return:
        visible_patch: boolean array of shape (2, 2k+1, 2k+1)
            1st channel: True if they are visible clear pixels
            2nd channel: True if they are visible wall pixels
        angle_ranges: list of tuples of feasible ranges of view that are still not blocked
    """
    if split_required:
        angle_ranges = split_angle_ranges(angle_ranges)
    h, w = square_patch.shape
    assert h == w and h % 2 == 1
    # map of visible pixels to be returned
    visible_patch = np.zeros((2, h, w), dtype=int)
    if square_patch.shape == (1, 1):
        visible_patch[0, 0, 0] = 1
        return visible_patch, angle_ranges
    # call function recursively to get visible patch for the inner (2k-1, 2k-1) patch, and update the angle_ranges
    visible_patch[:, 1:-1, 1:-1], angle_ranges = \
        get_partially_observable_pixels(square_patch[1:-1, 1:-1], angle_ranges, split_required=False)

    # get visible angle ranges after reaching the inner wall and update visible patch
    visible_patch, angle_ranges = update_visiblility(square_patch, visible_patch, angle_ranges, h / 2 - 1, w)
    # get visible angle ranges after reaching the outer wall and update visible patch
    visible_patch, angle_ranges = update_visiblility(square_patch, visible_patch, angle_ranges, h / 2, w)
    return visible_patch, angle_ranges


def update_visiblility(square_patch, visible_patch, angle_ranges, u, v):
    new_angle_range = []

    for ang_sm, ang_lg, side in angle_ranges:
        # rotate so that ang_sm and ang_lg is in range (-pi/4, pi/4)
        square_patch = rotate90(square_patch, -side)
        visible_patch = rotate90(visible_patch, -side)
        rotate = (pi / 2) * side

        # update strip visibility
        square_patch[-1, :], visible_patch[:, -1, :], strip_ang_ranges \
            = eval_strip_visibility(square_patch[-1, :], visible_patch[:, -1, :], ang_sm - rotate, ang_lg - rotate, u,
                                    v)

        # rotate back to original angle
        new_angle_range += [(s + rotate, l + rotate, side) for s, l in strip_ang_ranges]
        square_patch = rotate90(square_patch, side)
        visible_patch = rotate90(visible_patch, side)
    return visible_patch, new_angle_range


def eval_strip_visibility(map_strip, visible_strip, ang_sm, ang_lg, u, v):
    new_angle_range = []
    new_ang_sm = None
    for i in range(
            max(0, floor(v / 2 + u * tan(ang_sm))),
            min(int(v / 2 + u), ceil(v / 2 + u * tan(ang_lg)))
    ):
        if not map_strip[i]:
            # visible clear spaces
            visible_strip[0, i] = 1
            if new_ang_sm is None:
                new_ang_sm = max(ang_sm, atan((i - v / 2) / u))
        else:
            # visible walls
            visible_strip[1, i] = 1
            if new_ang_sm is not None:
                new_arg_lg = min(ang_lg, atan((i - v / 2) / u))
                new_angle_range.append((new_ang_sm, new_arg_lg))
                new_ang_sm = None

    if new_ang_sm is not None:
        new_angle_range.append((new_ang_sm, ang_lg))

    return map_strip, visible_strip, new_angle_range


def split_angle_ranges(angle_ranges):
    """
    make sure the ranges of (ang_sm, ang_lg) fit into (-pi/4, pi/4), (pi/4, 3pi/4), (3pi/4, 5pi/4), (5pi/4, 7pi/4)
    where each corresponds to the bottom, right, top, and left edges respectively
    :param angle_ranges: list of tuples of feasible ranges of view
        [(a1, b1), (a2, b2), ...] where (a1 < b1), (a2 < b2), ...
    :return: list of tuples of feasible ranges of view [(a1, b1, s1), (a2, b2, s2),...]
         where a < b and (a, b) fits into (-pi/4, pi/4), (pi/4, 3pi/4), (3pi/4, 5pi/4), (5pi/4, 7pi/4)
         and s denotes the side (0: bottom, 1: right, 2: top, 3: left)
    """
    adjusted_ranges = []
    while angle_ranges:
        ang_s, ang_l = angle_ranges.pop(0)
        assert 0 <= ang_l - ang_s <= 2 * pi
        # rotate both ang_s and ang_l by pi/4
        ang_s += pi / 4
        ang_l += pi / 4
        quadrant = ang_s // (pi / 2)
        # if ang_s and ang_l don't share quadrants, divide the range so that they do
        if ang_l // (pi / 2) > quadrant:
            next_quadrant = (quadrant + 1) * (pi / 2)
            # rotate next_ang_s and next_ang_l by -pi/4
            angle_ranges.append((next_quadrant - pi / 4, ang_l - pi / 4))
            ang_l = next_quadrant
        # normalize angle to ranges (-pi/4, 7pi/4)
        adj_ang_s = ang_s % (2 * pi) - pi / 4
        adj_ang_l = ang_l + adj_ang_s - ang_s
        adjusted_ranges.append((adj_ang_s, adj_ang_l, int(quadrant % 4)))

    return adjusted_ranges

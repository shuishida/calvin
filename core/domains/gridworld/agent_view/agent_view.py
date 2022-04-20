from math import pi

from core.domains.gridworld.agent_view.agent_view_utils import get_partially_observable_pixels, extract_view, embed_view


class AgentView:
    def __init__(self, view_range, view_angle=2*pi):
        self.view_range = view_range
        self.view_angle = view_angle
        self.N_CHANNELS = 2             # (0th channel: clear cells, 1st channel: obstacle cells)

    def local(self, square_patch, ang90=0):
        angle_ranges = self.get_angle_ranges(ang90)
        visible_patch, _ = get_partially_observable_pixels(square_patch, angle_ranges)
        return visible_patch

    def glob(self, grid, h, w, ang90=0):
        r = self.view_range
        angle_ranges = self.get_angle_ranges(ang90)
        square_patch = extract_view(grid, h, w, r)
        visible_patch, _ = get_partially_observable_pixels(square_patch, angle_ranges)
        return embed_view(visible_patch, grid.shape, h - r, w - r)

    def get_angle_ranges(self, ang90):
        """
        :param ang90: (0: bottom, 1: right, 2: top, 3: left)
        :return: angle range
        """
        center = pi / 2 * ang90
        return [(center - self.view_angle / 2, center + self.view_angle / 2)]

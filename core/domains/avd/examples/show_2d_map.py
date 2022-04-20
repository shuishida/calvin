import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')

from core.domains.avd.examples.manual_nav import manual_navigation
from core.domains.avd.dataset.data_classes import ImageNode
from core.domains.avd.dataset.scene_loader import get_scene_from_commandline


class Map2d:
    def __init__(self, map_x=1000, map_z=1000, y_thresh=0.5):
        self.map_x, self.map_z = map_x, map_z
        self.y_thresh = y_thresh
        self.map = np.zeros((map_x, map_z, 3), dtype=np.uint8)
        self.count = 0

    def __call__(self, image_node: ImageNode):
        points_np = image_node.point_cloud()
        for point in points_np:
            x, y, z, r, g, b = point
            if y < self.y_thresh:
                x = int(x * 50) + self.map_x // 2
                z = int(z * 50) + self.map_z // 2
                if 0 <= x < self.map_x and 0 <= z < self.map_z:
                    self.map[x, z] = np.array([int(r), int(g), int(b)])

        if self.count % 10 == 0:
            fig = plt.figure()
            plt.imshow(self.map)
            plt.draw()
            plt.waitforbuttonpress()
            plt.close()

        self.count += 1


if __name__ == "__main__":
    scene, args = get_scene_from_commandline()
    manual_navigation(scene, Map2d())

"""
python core/domains/avd/examples/show_2d_map.py -s Home_003_1
"""

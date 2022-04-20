import sys
import pptk
import numpy as np

sys.path.append('.')

from core.domains.avd.examples.manual_nav import manual_navigation
from core.domains.avd.dataset.data_classes import ImageNode
from core.domains.avd.dataset.scene_loader import get_scene_from_commandline


class Voxel:
    def __init__(self):
        self.voxel = {}
        self.count = 0

    def __call__(self, image_node: ImageNode):
        tmp_file = "tmp_point_cloud.ply"

        points_np = image_node.point_cloud()
        for point in points_np:
            x, y, z, r, g, b = point
            x = int(x * 100)
            y = int(y * 100)
            z = int(z * 100)
            self.voxel[(x, y, z)] = (r, g, b)

        if self.count % 10 == 0:

            xyz = []
            rgb = []
            for coord, color in self.voxel.items():
                xyz.append(list(coord))
                rgb.append(list(color))
            xyz = np.array(xyz)
            rgb = np.array(rgb) / 255
            v = pptk.viewer(xyz)
            v.attributes(rgb)
            v.set(point_size=0.1)

        self.count += 1


if __name__ == "__main__":
    scene, _ = get_scene_from_commandline()
    manual_navigation(scene, Voxel())


"""
python core/domains/avd/examples/show_voxel.py -s Home_003_1
"""

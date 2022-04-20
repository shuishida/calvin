import argparse
import sys
import numpy as np
import pptk

sys.path.append('.')

from core.domains.avd.dataset.data_classes import Scene
from core.domains.avd.dataset.scene_loader import get_scene_from_commandline


def get_point_cloud(scene: Scene, subsample=10):
    selected_points = []
    for i, image_node in enumerate(scene.image_nodes.values()):
        if i % subsample == 0:
            sys.stdout.write(f"\rComputing {i // subsample + 1} / {len(scene.image_nodes) // subsample} image nodes...")
            sys.stdout.flush()
            points = image_node.point_cloud()
            selected_points.append(points)
    print()
    selected_points = np.concatenate(selected_points)
    v = pptk.viewer(selected_points[:, :3])
    v.attributes(selected_points[:, 3:] / 255)
    v.set(point_size=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=150, type=int)
    scene, args = get_scene_from_commandline(parser)
    get_point_cloud(scene, args.sample)


"""
python core/domains/avd/examples/show_point_cloud.py -s Home_003_1
"""

import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.append('.')

from core.domains.avd.dataset.scene_loader import get_scene_from_commandline


def vis_objects(scene, target_name):
    fig, axes = plt.subplots(2, 1)
    ax_rgb, ax_mask = axes

    target_class = scene.object_classes.get_by_name(target_name)
    print(target_class)
    objects = []
    for obj in target_class:
        objects.append(obj)
        plt.cla()
        image = obj.image_node.rgb()
        H, W, _ = image.shape

        ax_rgb.imshow(image)
        x1, y1, x2, y2 = obj.bbox
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax_rgb.add_patch(rect)

        mask = np.zeros((H, W))
        mask[y1:y2, x1:x2] = 1
        ax_mask.imshow(mask)

        plt.draw()
        plt.waitforbuttonpress()

        for p in reversed(ax_rgb.patches):
            p.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default="coca_cola_glass_bottle", help="target name")
    scene, args = get_scene_from_commandline(parser)
    vis_objects(scene, target_name=args.target)

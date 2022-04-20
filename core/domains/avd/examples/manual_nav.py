import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


sys.path.append('.')

from utility.visualise.point_cloud import PointCloudO3D
from core.domains.avd.dataset.data_classes import Scene, ImageNode
from core.domains.avd.dataset.scene_loader import get_scene_from_commandline


def print_instructions():
    print("""Enter a character to move around the scene:
        'w' - forward
        'a' - rotate counter clockwise
        's' - backward
        'd' - rotate clockwise
        'e' - left
        'r' - right
        'q' - quit
        'h' - print this help menu""")


def manual_navigation(scene: Scene, callback=None):

    image_node: ImageNode = list(scene.image_nodes.values())[0]

    move_command = ''
    fig, axes = plt.subplots(2, 1)
    ax_rgb, ax_depth = axes

    print_instructions()

    while move_command != 'q':
        # load the current image and annotations
        rgb_image = image_node.rgb()
        depth_image = image_node.depth()
        objects = image_node.objects

        print(image_node.position)
        print(image_node.camera_direction)
        # plot the image and draw the boxes
        plt.cla()
        ax_rgb.imshow(rgb_image)
        depth_imshow = ax_depth.imshow(depth_image)
        ax_rgb.set_title(f"RGB: {image_node}")
        ax_depth.set_title(f"Depth: {image_node}")

        for instance in objects:
            # Create a Rectangle patch
            x1, y1, x2, y2 = instance.bbox
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax_rgb.add_patch(rect)

        cb = fig.colorbar(depth_imshow, ax=ax_depth)
        # draw the plot on the figure
        plt.draw()
        plt.pause(.001)

        if callback:
            callback(image_node)

        # get input from user
        move_command = input("Enter command ('h' for help, 'q' to quit): ")
        if move_command == 'h': print_instructions()
        else:
            # get the next image name to display based on the user input
            for move, next_node in image_node.moves.items():
                if move_command == move.value:
                    image_node = next_node
                    break
            else:
                print("This action wasn't available")

        # clean up
        cb.remove()
        for p in reversed(ax_rgb.patches):
            p.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc', action="store_true", help="show point clouds")

    scene, args = get_scene_from_commandline(parser)
    manual_navigation(scene, PointCloudO3D() if args.pc else None)


"""
python core/domains/avd/examples/manual_nav.py -s Home_003_1
"""

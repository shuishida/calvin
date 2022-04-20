import sys
import matplotlib.pyplot as plt

sys.path.append('.')

from core.domains.avd.dataset.data_classes import Scene
from core.domains.avd.dataset.scene_loader import get_scene_from_commandline


def show_camera_poses(scene: Scene, plot_directions=True, scale_positions=False, save_dir=None):
    fig = plt.figure()
    for image_node in scene.image_nodes.values():
        # get 3D camera position in the reconstruction coordinate frame. The scale is arbitrary.
        world_pos = image_node.position
        # get 3D vector that indicates camera viewing direction
        # add the world_pos to translate the vector from the origin to the camera location.
        direction = world_pos + image_node.camera_direction * 0.25

        if scale_positions:
            world_pos *= scene.scale
            direction *= scene.scale

        # plot only 2D, as all camera heights are the same
        # draw the position
        plt.plot(world_pos[0], world_pos[2], 'ro')
        # draw the direction if user sets option
        if plot_directions:
            plt.plot([world_pos[0], direction[0]],
                     [world_pos[2], direction[2]], 'b-')

            # for camera in image_structs
    plt.axis('equal')
    if save_dir is None:
        plt.show()
    else:
        fig.savefig(save_dir)


if __name__ == "__main__":
    scene, args = get_scene_from_commandline()
    show_camera_poses(scene)

"""
python core/domains/avd/examples/show_camera_poses.py -s Home_003_1
"""
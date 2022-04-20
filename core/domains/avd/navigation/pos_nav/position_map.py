from collections import OrderedDict
from typing import List, Dict, Set
import numpy as np
import sys
from einops import rearrange
import matplotlib.pyplot as plt
import torch

sys.path.append('.')

from core.domains.avd.navigation.pos_nav.actions import DirAction, action_to_move, action_to_dir
from core.domains.avd.dataset.const import AVDMove
from core.domains.avd.dataset.data_classes import Scene, ImageNode, ObjectClass
from scipy import spatial


class PositionMap:
    scene: Scene
    nodes: Dict[str, 'PositionNode']
    image_to_pos_node: Dict[str, 'PositionNode']
    kdtree = None

    def __init__(self, scene: Scene):
        self.scene = scene
        self.image_to_pos_node = {}
        self.nodes = OrderedDict()
        super(PositionMap, self).__init__()

    def add_pos_node(self, pos_node: 'PositionNode'):
        self.nodes[repr(pos_node)] = pos_node
        for node in pos_node.image_nodes:
            self.image_to_pos_node[node.image_name] = pos_node

    def init_kdtree(self):
        self.kdtree = spatial.KDTree([node.position for node in self.nodes.values()])

    def query(self, position):
        dist, index = self.kdtree.query(position)
        return dist, list(self.nodes.values())[index]


def get_node_image_names(pos_node_name: str):
    return pos_node_name.split("__")


class PositionNode:
    image_nodes: List[ImageNode]
    neighbour_nodes: List[ImageNode]
    transitions: Dict[DirAction, 'PositionNode']
    position: np.ndarray
    objects: Set[ObjectClass]
    pos_map: PositionMap

    def __init__(self, pos_map: PositionMap, image_nodes: List[ImageNode], neighbour_nodes: List[ImageNode]):
        self.image_nodes = image_nodes
        self.position = np.concatenate([[node.position] for node in image_nodes], axis=0).mean(
            axis=0).reshape(-1)
        self.neighbour_nodes = neighbour_nodes
        self.neighbour_dirs = [self.get_neighbour_dir(node) for node in neighbour_nodes]
        self.kdtree = spatial.KDTree(self.neighbour_dirs) if self.neighbour_dirs else None
        self.pos_map = pos_map
        self.transitions = {}
        self.objects = set(o.object_class for node in image_nodes for o in node.objects)

    def __repr__(self):
        return "__".join(sorted([n.image_name for n in self.image_nodes]))

    def get_neighbour_dir(self, neighbour):
        vec = neighbour.position.reshape(-1) - self.position
        return vec / np.linalg.norm(vec)

    def set_transitions(self):
        if self.kdtree is None: return
        for action in DirAction:
            if action == DirAction.COMPLETE: continue
            action_dir = action_to_dir(action)
            _, index = self.kdtree.query(action_dir)
            inner_prod = np.inner(self.neighbour_dirs[index], action_dir)
            if inner_prod >= np.sqrt(2) / 2:
                self.transitions[action] = self.pos_map.image_to_pos_node[self.neighbour_nodes[index].image_name]

    def point_cloud(self):
        """
        :return: (N, 6) numpy array where each row is (X, Y, Z, R, G, B)
        """
        return np.concatenate([node.point_cloud() for node in self.image_nodes])

    @property
    def is_valid_coords(self):
        """
        :return: (12, H, W) numpy array
        """
        return np.stack([node.is_valid_coords for node in self.image_nodes])

    def rgb(self):
        """
        :return: (12, H, W, 3) numpy array
        """
        return np.stack([node.rgb() for node in self.image_nodes])

    def normalised_rgbd(self):
        """
        :return: (12, H, W, 4) numpy array
        """
        return np.stack([node.normalised_rgbd() for node in self.image_nodes])

    @property
    def world_coords(self):
        """
        :return: (12, H, W, 3) numpy array
        """
        return np.stack([node.world_coords for node in self.image_nodes])

    def sample_free_space(self, n_samples_per_pixel=1):
        """
        :return: (12, H, W, k, 3) numpy array where last row is (X, Y, Z)
        """
        return np.stack([node.sample_free_space(n_samples_per_pixel)
                         for node in self.image_nodes])

    def embedding(self):
        return torch.stack([node.embedding() for node in self.image_nodes])


def create_position_map(scene: Scene, interval=1):
    pos_map = PositionMap(scene)
    for i, image_node in enumerate(scene.image_nodes.values()):
        if pos_map.image_to_pos_node.get(image_node.image_name): continue
        # make list of nodes which share the same position
        nodes = []
        curr_node = image_node
        neighbours = set()
        # check out node connected counter clockwise until you come back to the initial node
        while True:
            nodes.append(curr_node)
            neighbours |= set(curr_node.moves.values())
            curr_node = curr_node.moves[AVDMove.rotate_ccw]
            if curr_node == image_node: break
        neighbours -= set(nodes)
        pos_node = PositionNode(pos_map, nodes[::interval], list(neighbours))
        pos_map.add_pos_node(pos_node)
    pos_map.init_kdtree()
    for pos_node in pos_map.nodes.values():
        pos_node.set_transitions()
    return pos_map


def show_position_map(pos_nodes: List[PositionNode], save_dir=None):
    fig = plt.figure()
    for i, pos_node in enumerate(pos_nodes):
        curr_pos = pos_node.position
        plt.plot(curr_pos[0], curr_pos[2], 'ro')

        for j, neighbour in enumerate(pos_node.transitions.values()):
            sys.stdout.write(f"\r{i} / {len(pos_nodes)}, {j} / {len(pos_node.transitions)}")
            sys.stdout.flush()
            next_pos = neighbour.position
            plt.plot([curr_pos[0], next_pos[0]],
                     [curr_pos[2], next_pos[2]], 'b-')
    plt.axis('equal')
    if save_dir is None:
        plt.draw()
        plt.waitforbuttonpress()
    else:
        fig.savefig(save_dir)
    plt.close()


# if __name__ == "__main__":
#     scene, _ = get_scene_from_commandline()
#     pos_map = create_position_map(scene)
#     show_position_map(pos_map.nodes)
#     show_point_cloud(pos_map.nodes)

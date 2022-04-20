import numpy as np

from core.domains.avd.navigation.planner import AVDPlannerBase, AVDMDPMetaBase
from core.domains.avd.navigation.pos_nav.position_map import PositionNode, create_position_map


class AVDPosMDPMeta(AVDMDPMetaBase):
    def __init__(self, *args, **kwargs):
        super(AVDPosMDPMeta, self).__init__(*args, **kwargs)
        self.maps = {}
        for scene_name, scene in self.scenes.items():
            pos_map = create_position_map(scene, interval=2)
            self.maps[scene_name] = pos_map

    def get_states_from_scene(self, scene_name):
        pos_map = self.maps[scene_name]
        return list(pos_map.nodes.values())

    def get_states_from_objects(self, objects):
        return [self.maps[obj.image_node.scene.name].image_to_pos_node[obj.image_node.image_name] for obj in objects]

    def state_to_index(self, state: PositionNode):
        h, _, w = state.position
        h1, w1, h2, w2 = self.map_bbox
        assert h1 <= h < h2, f"h dimension {h} out of range [{h1}, {h2})"
        assert w1 <= w < w2, f"w dimension {w} out of range [{w1}, {w2})"
        x_size, z_size = self.state_shape
        x_ind = int((h - h1) / (h2 - h1) * x_size)
        z_ind = int((w - w1) / (w2 - w1) * z_size)
        return x_ind, z_ind

    def state_index_to_grid_index(self, state_index):
        return state_index


class AVDPosPlanner(AVDPlannerBase):
    def transition(self, curr_state: PositionNode, action, reverse=False):
        """
        :param curr_state:
        :param action:
        :param reverse: if reversed, give previous state instead of next state
        :return: list of (next_state, action, cost) tuples, (prev_state, action, cost) if reversed
        """
        if action == self.meta.actions.done: return None
        if reverse:
            raise NotImplementedError
        new_state = curr_state.transitions.get(action)
        if new_state:
            cost = np.linalg.norm(new_state.position - curr_state.position)
            return new_state, action, cost
        return None

import os

import numpy as np

from core.domains.avd.dataset.scene_manager import AVDSceneManager
from core.domains.avd.navigation.planner import AVDPlannerBase, AVDMDPMetaBase
from core.domains.avd.navigation.pose_nav.actions import AVDPoseActionSet
from core.domains.avd.dataset.data_classes import ImageNode


class AVDPoseMDPMeta(AVDMDPMetaBase):
    def get_states_from_scene(self, scene_name):
        scene = self.scenes[scene_name]
        return list(scene.image_nodes.values())

    def get_states_from_objects(self, objects):
        return [obj.image_node for obj in objects]

    def state_to_index(self, state: ImageNode):
        pose = self.state_to_pose(state)
        return self.pose_to_index(pose)

    def state_index_to_grid_index(self, state_index):
        return state_index[1:]

    def pos_to_grid_index(self, pos):
        h, _, w = pos
        h1, w1, h2, w2 = self.map_bbox
        assert h1 <= h < h2, f"h dimension {h} out of range [{h1}, {h2})"
        assert w1 <= w < w2, f"w dimension {w} out of range [{w1}, {w2})"
        h_size, w_size = self.map_res
        h_ind = int((h - h1) / (h2 - h1) * h_size)
        w_ind = int((w - w1) / (w2 - w1) * w_size)
        return h_ind, w_ind

    def _vec_to_dir(self, vec):
        h, _, w = vec
        if h == 0: return 0 if w > 0 else -np.pi
        if h > 0: return np.arctan(w / h) - np.pi / 2
        return np.arctan(w / h) + np.pi / 2

    def dir_to_ori_index(self, d):
        return int((self.ori_res * d / (2 * np.pi) + 0.5) % self.ori_res)

    def state_to_pose(self, state):
        h, _, w = state.position
        d = self._vec_to_dir(state.camera_direction)
        return d, h, w

    def pose_to_index(self, pose):
        d, h, w = pose
        h, w = self.pos_to_grid_index((h, 0, w))
        o = self.dir_to_ori_index(d)
        return o, h, w


class AVDPosePlanner(AVDPlannerBase):
    def transition(self, curr_state: ImageNode, action: AVDPoseActionSet, reverse=False):
        """
        :param curr_state:
        :param action:
        :param reverse: if reversed, give previous state instead of next state
        :return: list of (next_state, action, cost) tuples, (prev_state, action, cost) if reversed
        """
        meta = self.meta
        if reverse:
            raise Exception("Reverse transition not implemented")
        if action not in [meta.actions.move_forward, meta.actions.turn_right, meta.actions.turn_left]: return None
        new_state = curr_state.moves.get(action)
        if new_state:
            cost = np.linalg.norm(new_state.position - curr_state.position)
            return new_state, action, cost
        return None

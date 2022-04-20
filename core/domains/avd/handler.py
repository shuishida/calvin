from typing import Tuple, List

import torch
import numpy as np
from einops import rearrange

from core.domains.avd.dataset.data_classes import Scene
from core.domains.avd.navigation.pos_nav.position_map import PositionNode, get_node_image_names
from core.handler import VisNavHandler, dict_collate


class AVDDataHandler(VisNavHandler):
    def postproc_obsvs(self, obsvs, episode_info) -> dict:
        return obsvs

    @property
    def obsv_keys(self):
        return ['state_info', 'poses']

    def combine_info(self, curr_info, past_seq_info, future_seq_info, full_seq_info, step, inference: bool):
        meta = self.meta
        scene_name = curr_info['scene_name']
        scene: Scene = meta.scenes[scene_name]
        if meta.ori_res:
            image_names = past_seq_info.get('state_info', [])
        else:
            image_names = []
            for node_name in past_seq_info.get('state_info', []):
                image_names += get_node_image_names(node_name)

        indices = scene.names_to_indices(image_names)
        rgb = scene.rgb_images[indices]
        depth = scene.depth_images[indices] / scene.scale
        emb = scene.embeddings[indices]
        valids = depth != 0
        coords = scene.coords(image_names)

        if meta.ori_res:
            curr_rgb = scene.image_nodes[curr_info['curr_state_info']].rgb()

            rgb = rearrange(rgb, "b h w f -> b f h w")
            coords = rearrange(coords, "b h w f -> b f h w")
        else:
            curr_node_name = curr_info['curr_state_info']
            curr_image_names = get_node_image_names(curr_node_name)
            curr_indices = scene.names_to_indices(curr_image_names)
            curr_rgb = rearrange(scene.rgb_images[curr_indices], "o h w f -> (o h) w f")
            n_ori = len(curr_image_names)

            rgb = rearrange(rgb, "(b o) h w f -> b f (o h) w", o=n_ori)
            emb = rearrange(emb, "(b o) f h w -> b f (o h) w", o=n_ori)
            valids = rearrange(valids, "(b o) h w -> b (o h) w", o=n_ori)
            coords = rearrange(coords, "(b o) h w f -> b f (o h) w", o=n_ori)

        return {
            **curr_info,
            'curr_rgb': rearrange(torch.from_numpy(curr_rgb).float() / 255, "h w f -> f h w")
        }, {
            **past_seq_info,
            "rgb": torch.from_numpy(rgb).float() / 255,
            "emb": emb,
            "surf_xyz": torch.from_numpy(coords).float(),
            "valid_points": torch.from_numpy(valids).bool()
        }, future_seq_info, full_seq_info

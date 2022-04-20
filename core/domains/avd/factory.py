import os

from core.domains.avd.dataset.scene_manager import AVDSceneManager
from core.domains.avd.env import AVDEnv
from core.domains.avd.navigation.pos_nav.actions import AVDPosActionSet
from core.domains.avd.navigation.pos_nav.pos_planner import AVDPosMDPMeta
from core.domains.avd.navigation.pose_nav.actions import AVDPoseActionSet
from core.domains.avd.navigation.pose_nav.pose_planner import AVDPoseMDPMeta


def init_avd_meta(avd_data: str = None, map_bbox: tuple = None, map_res: tuple = None, ori_res=None, resize: tuple = None,
                  target=None, in_ram=None, target_size_ratio=None, target_dist_thresh=None, avd_workers=None, **kwargs) -> AVDPoseMDPMeta:
    scenes = AVDSceneManager(os.path.join(avd_data, "src"), resize, target, in_ram, target_size_ratio, target_dist_thresh, avd_workers)
    return AVDPoseMDPMeta(scenes, map_bbox, map_res, ori_res) if ori_res else AVDPosMDPMeta(scenes, map_bbox, map_res)


def init_avd_env(meta, **kwargs):
    return AVDEnv(meta, **kwargs)


def get_avd_model_config(model_config, map_bbox=None, map_res=None, ori_res=None, **kwargs):
    model_config = dict(model_config)
    action_set = AVDPoseActionSet() if ori_res else AVDPosActionSet()
    model_config['action_set'] = action_set
    model_config['xyz_to_h'] = 0
    model_config['xyz_to_w'] = 2
    model_config['l_i'] = model_config['pcn_f'] + model_config['v_res']
    model_config['ori_res'] = ori_res
    model_config['map_res'] = map_res
    model_config['map_bbox'] = map_bbox
    model_config['use_embeddings'] = True
    model_config['pcn_i'] = 128
    return model_config

from core.domains.miniworld.env import MiniWorldEnv
from core.domains.miniworld.gym_envs.maze import MiniMaze
from core.utils.utils import set_random_seed
from core.mdp.actions import OmniActionSet
from core.domains.gridworld.actions import GridDirs
from core.domains.miniworld.actions import MiniWorldActionSet
from core.domains.miniworld.planner import MiniWorldMDPMeta


def init_miniworld_meta(map_bbox=None, map_res=None, ori_res=None, **kwargs) -> MiniWorldMDPMeta:
    return MiniWorldMDPMeta(map_bbox, map_res, ori_res)


def init_miniworld_env(meta, *, seed=None, i_env=None, size=None, min_traj_len=0, costmap_margin=None, max_steps=None,
                       sample_free=None, full_view=None, **kwargs):
    env = MiniMaze(num_rows=size, num_cols=size, max_steps=max_steps)
    if seed:
        print(f"Setting random seed for env {i_env} to {seed + i_env}")
        env.seed(seed + i_env)

    print("min trajectory length: ", min_traj_len)

    return MiniWorldEnv(meta, env, min_traj_len=min_traj_len,
                        costmap_margin=costmap_margin, max_steps=max_steps, sample_free=sample_free, full_view=full_view)


def get_miniworld_model_config(model_config, map_bbox=None, map_res=None, ori_res=None, **kwargs):
    model_config = dict(model_config)
    model_config['action_set'] = MiniWorldActionSet() if ori_res else OmniActionSet(ori_res)
    model_config['dirs'] = GridDirs(ori_res == 4) if ori_res in [4, 8] else None
    model_config['xyz_to_h'] = 2
    model_config['xyz_to_w'] = 0
    model_config['l_i'] = model_config['pcn_f'] + model_config['v_res']
    model_config['ori_res'] = ori_res
    model_config['map_res'] = map_res
    model_config['map_bbox'] = map_bbox
    return model_config

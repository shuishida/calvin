from core.domains.gridworld.env import GridEnv
from core.domains.gridworld.planner.gridmap_planner import GridMDPMeta
from core.utils.utils import set_random_seed
from core.mdp.actions import EgoActionSet
from core.mdp.meta import MDPMeta
from core.domains.gridworld.actions import GridDirs, GridActionSet
from core.domains.gridworld.map.parse_gridmap import get_map
from core.domains.gridworld.planner.ego_grid_planner import EgoGridMDPMeta


def init_grid_meta(size=None, four_way=False, ego=False, **kwargs) -> MDPMeta:
    return EgoGridMDPMeta(size, four_way) if ego else GridMDPMeta(size, four_way)


def init_grid_env(meta, *, map=None, min_traj_len=0, ego=False, view_range=0,
                  target_known=False, allow_backward=True, max_steps=None, **map_args):
    gridmap = get_map(map, **map_args)
    return GridEnv(meta, gridmap, min_traj_len=min_traj_len, ego=ego, view_range=view_range,
                   allow_backward=allow_backward, target_known=target_known, max_steps=max_steps)


def get_grid_env_model_config(model_config, ego=False, four_way=False, **kwargs):
    model_config = dict(model_config)
    if ego:
        model_config['action_set'] = EgoActionSet()
        model_config['ori_res'] = 8
        model_config['dirs'] = GridDirs(four_way)
    else:
        model_config['action_set'] = GridActionSet(four_way)
    return model_config

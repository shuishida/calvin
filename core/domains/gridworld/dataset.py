import argparse
import os
import sys

sys.path.append('.')

from core.dataset import add_demo_gen_args, generate_expert_demos
from core.domains.gridworld.map.parse_gridmap import get_map, add_map_args


def get_save_path(data=None, n_episodes=None, map=None, view_range=None, min_traj_len=None,
                  four_way=False, ego=False, max_steps=None, **kwargs):
    return os.path.join(data, f"{repr(get_map(map, **kwargs))}"
                              f"{'' if view_range is None else '_vr_' + str(view_range)}"
                              f"_{n_episodes}"
                              f"_{min_traj_len}"
                              f"_{max_steps}"
                              f"{'_4way' if four_way else ''}"
                              f"{'_ego' if ego else ''}")


def generate_grid_expert_demos(config):
    config['allow_backward'] = False
    config['episode_in_mem'] = True
    config['obsv_in_mem'] = True
    config['trans_in_mem'] = True
    config['domain'] = 'grid'
    generate_expert_demos(get_save_path, **config)


def add_gridworld_env_args(parser: argparse.ArgumentParser):
    add_map_args(parser)
    parser.add_argument('--min_traj_len', '-trajlen', required=True, type=int,
                        help="minimum threshold of trajectory length")
    parser.add_argument('--ego', action='store_true', default=False)
    parser.add_argument('--four_way', '-4', action='store_true', default=False)
    parser.add_argument('--view_range', '-vr', default=None, type=int, help="view range of the agent")
    parser.add_argument('--max_steps', '-mxs', default=None, type=int, help="max steps before end of episode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/gridworld", help="root path to save data")
    add_demo_gen_args(parser)
    add_gridworld_env_args(parser)
    config = parser.parse_args()

    generate_grid_expert_demos(vars(config))


if __name__ == "__main__":
    main()

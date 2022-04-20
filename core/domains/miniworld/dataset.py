import sys
import argparse
import os

sys.path.append('.')

from core.dataset import add_demo_gen_args, generate_expert_demos


def get_save_path(data=None, size=None, n_episodes=None, map_bbox=None,
                  map_res=None, ori_res=None, min_traj_len=None, max_steps=None, sample_free=None,
                  full_view=None, **kwargs):
    return os.path.join(data, f"Maze_{size}__{'_'.join(map(str, map_bbox))}__{'_'.join(map(str, map_res))}__{ori_res or 'pos'}"
                              f"_{n_episodes}_{min_traj_len}_{max_steps}_{sample_free}{'_fv' if full_view else ''}")


def generate_miniworld_expert_demos(config):
    config['episode_in_mem'] = True
    config['obsv_in_mem'] = False
    config['trans_in_mem'] = True
    config['domain'] = 'miniworld'
    generate_expert_demos(get_save_path, **config)


def add_miniworld_env_args(parser: argparse.ArgumentParser):
    parser.add_argument('--size', '-sz', type=int, default=3, help="MiniWorld maze size")
    parser.add_argument("--map_bbox", "-bbox", help="bounding box for map (h1, w1, h2, w2)",
                        type=int, nargs=4, default=(0, 0, 10, 10))
    parser.add_argument("--map_res", "-res", help="map resolution",
                        type=int, nargs=2, default=(30, 30))
    parser.add_argument("--ori_res", '-ori', type=int, default=None, help="orientation resolution")
    parser.add_argument("--costmap_margin", type=int, default=5, help="costmap margin")
    parser.add_argument('--min_traj_len', '-trajlen', type=int, default=0, help="minimum threshold of trajectory length")
    parser.add_argument('--max_steps', '-mxs', default=None, type=int, help="max steps before end of episode")
    parser.add_argument("--sample_free", help="number of free space samples per pixel",
                        type=int, default=8)
    parser.add_argument('--full_view', '-fv', action="store_true", help="full 360 degrees view")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/miniworld", help="root path to save data")
    add_demo_gen_args(parser)
    add_miniworld_env_args(parser)
    config = parser.parse_args()

    generate_miniworld_expert_demos(vars(config))


if __name__ == "__main__":
    main()

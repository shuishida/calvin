import argparse
import os
import sys

sys.path.append('.')

from core.dataset import add_demo_gen_args, generate_expert_demos


def get_save_path(data=None, n_episodes=None, map_bbox=None, map_res=None, ori_res=None, resize=None,
                  target=None, min_traj_len=None, max_steps=None, sample_free=None, **kwargs):
    return os.path.join(data, "pose_nav" if ori_res else "pos_nav",
                        f"{n_episodes}_{'_'.join(map(str, map_bbox))}__{'_'.join(map(str, map_res))}__{ori_res or 'pos'}"
                        f"_{'_'.join(map(str, resize))}_{target}_{min_traj_len}_{max_steps}_{sample_free}")


def generate_avd_expert_demos(config):
    config['episode_in_mem'] = True
    config['obsv_in_mem'] = True
    config['trans_in_mem'] = True
    config['in_ram'] = True
    config['domain'] = 'avd'
    config['avd_data'] = config['data']
    generate_expert_demos(get_save_path, **config)


def add_avd_env_args(parser: argparse.ArgumentParser):
    parser.add_argument("--avd_workers", type=int, default=8, help="store embeddings in ram")
    parser.add_argument("--map_bbox", "-bbox", help="bounding box for map (h1, w1, h2, w2)",
                        type=int, nargs=4, default=(-15, -15, 15, 15))
    parser.add_argument("--map_res", "-res", help="map resolution",
                        type=int, nargs=2, default=(40, 40))
    parser.add_argument("--ori_res", "-ori", type=int, default=None, help="orientation resolution")
    parser.add_argument("--resize", "-imsize", help="resize images", type=int, nargs=2, default=None)
    parser.add_argument('--target', default=None, help="target name")
    parser.add_argument('--min_traj_len', '-trajlen', required=True, type=int,
                        help="minimum threshold of trajectory length")
    parser.add_argument('--target_size_ratio', '-tsr', default=0.6, type=float, help="target size ratio")
    parser.add_argument('--max_steps', '-mxs', default=None, type=int, help="max steps before end of episode")
    parser.add_argument("--sample_free", help="number of free space samples per pixel", type=int, default=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/avd", help="root path to save data")
    add_demo_gen_args(parser)
    add_avd_env_args(parser)
    config = parser.parse_args()

    generate_avd_expert_demos(vars(config))


if __name__ == "__main__":
    main()

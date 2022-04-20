import argparse
import os
import sys

sys.path.append(".")

from core.utils.trainer_utils import setup_agent
from core.experiences import ExperienceManager


def eval_agent(data=None, name=None, checkpoint=None, **config):
    env_config, meta, handler, trainer, agent, init_env = \
        setup_agent(data=data, checkpoint=checkpoint, **config)

    original_dir = os.path.dirname(checkpoint)
    save_dir = original_dir + "_" + name

    experiences = ExperienceManager(handler, save_dir=save_dir, **{**env_config, 'clear': True})

    env = init_env(meta, **env_config)
    eval_stats, eval_dur = agent.rollout(env, experiences=experiences, train=False, manual=True)

    print(eval_stats)


def add_eval_args(parser):
    parser.add_argument("checkpoint", help="checkpoint")
    parser.add_argument("--data", default=None, help="path to data directory")
    parser.add_argument("--name", default="manual", help="name of save directory")
    parser.add_argument("--device", "-d", default="cuda", help="device (cuda, cuda:0, cpu, etc.)")
    return parser


def main():
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    add_eval_args(parser)
    config = vars(parser.parse_args())

    checkpoint = config['checkpoint']
    if not config['data']:
        config['data'] = os.path.join(os.path.dirname(checkpoint), "..", "..", "..")
    eval_agent(**config)


if __name__ == "__main__":
    main()

"""
python core/scripts/manual.py data/gridworld/MazeMap_15x15_vr_2_4000_15_ego/models/.../
"""

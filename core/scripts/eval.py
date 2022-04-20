import argparse
import os
import shutil
import sys

sys.path.append(".")

from core.utils.trainer_utils import setup_agent
from core.env import VecEnv
from core.experiences import ExperienceManager


def eval_agent(data=None, name=None, checkpoint=None, n_envs=None, n_evals=None, split=None, **config):
    env_config, meta, handler, trainer, agent, init_env = \
        setup_agent(data=data, checkpoint=checkpoint, **config)

    vecenv = VecEnv([lambda: init_env(meta, i_env=i_env, split=split, **env_config) for i_env in range(n_envs)])

    original_dir = os.path.dirname(checkpoint)
    save_dir = original_dir + "_" + name

    experiences = ExperienceManager(handler, save_dir=save_dir, **{**env_config, 'clear': True}, cash_size=n_envs * 2)
    eval_stats, eval_dur = agent.rollouts(vecenv, n_episodes=n_evals, train=False,
                                          experiences=experiences, save_pred=True)

    print(eval_stats)

    shutil.copyfile(checkpoint, os.path.join(save_dir, "checkpoint.pt"))
    shutil.copyfile(checkpoint + ".json", os.path.join(save_dir, "checkpoint.pt.json"))
    shutil.copyfile(os.path.join(save_dir, "..", "..", "..", "env_config.json"),
                    os.path.join(save_dir, "env_config.json"))


def add_eval_args(parser):
    parser.add_argument("checkpoint", help="checkpoint")
    parser.add_argument("--data", default=None, help="path to data directory")
    parser.add_argument("--name", default="eval", help="name of save directory")
    parser.add_argument("--split", default="val", help="train / val split")
    parser.add_argument("--device", "-d", default="cuda", help="device (cuda, cuda:0, cpu, etc.)")
    parser.add_argument("--n_evals", "-nev", type=int, help="number of evaluation runs", default=100)
    parser.add_argument("--n_envs", "-env", type=int, help="number of envs", default=3)
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
python core/scripts/eval.py data/gridworld/MazeMap_15x15_vr_2_4000_15_ego/models/.../ -nev 5
"""

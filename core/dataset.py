import argparse
import json
import os

from core.domains.factory import get_factory
from core.experiences import ExperienceManager
from core.utils.utils import set_random_seed


def generate_expert_demos(get_save_path, data: str = None, n_episodes: int = 1000, clear: bool = False,
                          val_ratio: float = 0.25, **env_config):
    set_random_seed(env_config['seed'])

    domain = env_config['domain']
    factory = get_factory(domain)
    meta = factory.meta(data=data, **env_config)
    handler = factory.handler(meta, data=data, **env_config)
    save_dir = get_save_path(data=data, n_episodes=n_episodes, **env_config)

    print("Generating training data...")
    env = factory.env(meta, data=data, split="train", **env_config)
    train_expert_demos = ExperienceManager(handler, save_dir=os.path.join(save_dir, "train"), cash_size=1, clear=clear, **env_config)
    train_expert_demos.collect_demos(env, n_episodes=n_episodes)
    env.close()

    print("Generating validation data...")
    env = factory.env(meta, split="val", **env_config)
    val_expert_demos = ExperienceManager(handler, save_dir=os.path.join(save_dir, "val"), cash_size=1, clear=clear, **env_config)
    val_expert_demos.collect_demos(env, n_episodes=int(n_episodes * val_ratio))
    env.close()

    with open(os.path.join(save_dir, "env_config.json"), "w") as f:
        json.dump(env_config, f)


def add_demo_gen_args(parser: argparse.ArgumentParser):
    parser.add_argument('--clear', action="store_true", help="renew dataset")
    parser.add_argument('--n_episodes', '-n', default=1000, type=int, help="number of expert episodes")
    parser.add_argument('--val_ratio', '-val', default=0.25, type=float,
                        help="ratio of validation of trajectory samples")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

import argparse
import json
import os
import shutil
import sys
import torch

sys.path.append(".")

from core.domains.aihabitat.plot import habitat_plot
from core.domains.miniworld.plot import miniworld_plot, mini_rollout_plot, ego_mini_rollout_plot
from core.domains.factory import get_factory
from core.domains.gridworld.plot import grid_rollout_plot, ego_grid_plot, grid_plot
from core.experiences import ExperienceManager
from core.utils.tensor_utils import to_numpy


def get_visualiser_from_name(name):
    available = [grid_rollout_plot, ego_grid_plot, miniworld_plot, grid_plot, habitat_plot, mini_rollout_plot, ego_mini_rollout_plot]
    vis_dict = {vis.__name__: vis for vis in available}
    return vis_dict[name + "_plot"]


def visualise_experience(env_path, data_path, visualiser, show=False, save=False):
    with open(env_path, "r") as f:
        env_config = json.load(f)

    with open(os.path.join(data_path, "checkpoint.pt.json"), "r") as f:
        config = json.load(f)

    env_config = {**env_config, **config}

    domain = env_config['domain']
    factory = get_factory(domain)
    meta = factory.meta(**env_config)
    handler = factory.handler(meta, **env_config)
    model_config = factory.model_config(config, **env_config)
    experiences = ExperienceManager(handler, save_dir=data_path, **env_config)

    print([e.index for e in experiences.active_episodes])
    for i, episode in enumerate(experiences.active_episodes):
        episode_dir = os.path.join(data_path, f"episode_{i:03}")
        if os.path.exists(episode_dir): shutil.rmtree(episode_dir)
        os.makedirs(episode_dir)
        for j in range(len(episode)):
            epi, past_seq, full_seq, future_seq, pred = episode.get(j, inference=True, include_pred=True)
            data = {**epi, **past_seq, **full_seq, **future_seq, **pred}
            to_numpy(data)
            if save:
                torch.save(data, os.path.join(episode_dir, f"step_{j:04}.pt"))
            visualiser(meta, **{**data, **model_config}, save_path=os.path.join(episode_dir, f"step_{j:04}.png"),
                       show=show)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Experience directory")
    parser.add_argument("name", help="Plotter name")
    # parser.add_argument("--inference", '-inf', action='store_true', help="Plot inference data")
    parser.add_argument("--show", action='store_true', help="Show plot")
    parser.add_argument("--save", action='store_true', help="Save inputs and outputs")
    config = parser.parse_args()

    data_path = config.data

    env_path = os.path.join(data_path, "env_config.json")

    visualise_experience(env_path, data_path, get_visualiser_from_name(config.name), config.show, config.save)


if __name__ == '__main__':
    main()

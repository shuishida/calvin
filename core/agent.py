import sys
import time
from typing import List, Union, Any, Tuple

import os
import torch

from core.env import Env, VecEnv
from core.experiences import ExperienceManager, Episode
from core.handler import DataHandler
from core.trainer import Trainer
from core.utils.tensor_utils import to_numpy
from core.utils.utils import Stats


class Agent:
    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError

    def save(self):
        pass

    def reset(self, episode_infos: Union[List[Tuple[dict, Any]], Tuple[dict, Any]], resets: List[bool] = None):
        """
        Reset is called before each episode begins
        Leave blank if nothing needs to happen there
        :param resets: which episodes to reset (in case of multiple environments). Leave it as None if you want to reset all.
        :param t the number of timesteps in the episode
        """
        if not isinstance(episode_infos, list): episode_infos = [episode_infos]
        if resets is not None:
            assert len(episode_infos) == len(resets), "mask should have the same length as the number of envs"
        return self._reset(episode_infos, resets)

    def _reset(self, episode_infos: List[Tuple[dict, Any]], mask: List[bool] = None, **kwargs):
        raise NotImplementedError

    def step(self, episodes: List[Episode], save_pred=False) -> Union[List[Any], Any]:
        is_multi_input = False
        if isinstance(episodes, list):
            is_multi_input = True
        else:
            episodes = [episodes]

        actions = self._step(episodes, save_pred=save_pred)
        return actions if is_multi_input else actions[0]

    def _step(self, episodes, **kwargs):
        raise NotImplementedError

"""
meta = Meta()
env = Env(meta)
handler = DataHandler(meta)
experiences = ExperienceManager(handler, "experiences")
expert_demos = ExperienceManager(handler, "demos")
agent = Agent(experiences, expert_demos)

for i_episode in range(n_episodes):
    obsv, info = env.reset()
    reward = 0
    done = False

    episode_info = env.episode_info()
    agent.reset(episode_info)

    while not done:
        action = agent.step(obsv, reward, done, info)
        obsv, reward, done, info = env.step(action)
    
    agent.done(obsv, reward, info)
"""


class MemoryAgent(Agent):
    def __init__(self, handler: DataHandler, trainer: Trainer, experiences: ExperienceManager = None):
        self.handler = handler
        self.experiences = experiences
        self.trainer = trainer

        self.obsvs = None
        self.actions = None
        self.carrier = None
        self.count_step = 0
        self.active_episodes: List[Episode] = []

        self.training = True

    def _reset(self, episode_infos: List[Tuple[dict, Any]], resets: List[bool] = None,
               experiences: ExperienceManager = None, **kwargs):
        """
        :param episode_infos: either a dictionary of episode-specific info, or a list of episode info.
        :param resets: which episodes to reset (in case of multiple environments). Leave it as None if you want to reset all.
        :param t:
        :return:
        """
        if experiences is None:
            assert self.experiences is not None, "experience collector not set for this agent"
            experiences = self.experiences
        if resets is None or not self.active_episodes:
            self.active_episodes = [None] * len(episode_infos)
            self.actions = [None] * len(episode_infos)
            resets = [True] * len(episode_infos)
        new_episodes = []
        for i, ((episode_info, init_obsv), reset) in enumerate(zip(episode_infos, resets)):
            if reset:
                self.active_episodes[i] = episode = experiences.add_episode({**episode_info, "is_expert": False}, init_obsv)
                self.actions[i] = None
                new_episodes.append(episode)
        return new_episodes

    def _step(self, episodes: List[Episode], save_pred=False, **kwargs):
        """
        :param obsvs: agent's observation of the current environment
        :param rewards: amount of reward returned after previous action
        :param dones: whether the episode has ended.
        :param infos: extra information about the environment.
        :return: the actions to take
        """
        processed_obsvs = [episode.get(inference=True) for episode in episodes]
        new_episodes = [len(episode) == 0 for episode in episodes]

        histories = self.handler.collate(processed_obsvs)

        self.actions, outputs, self.carrier = self.policy(histories, new_episodes, self.carrier)

        if save_pred:
            preds = self.handler.postproc_preds(outputs)
            for episode, pred in zip(self.active_episodes, preds):
                episode.push_pred(pred)

        self.count_step += len(self.active_episodes)

        return self.actions

    def rollouts(self, env: VecEnv, n_steps: int = None, n_episodes: int = None,
                 train: bool = True, experiences: ExperienceManager = None, save_pred=False):
        start_time = time.time()
        stats = Stats()
        if experiences is None: experiences = self.experiences
        self.training = train
        actions = None
        count_steps = count_episodes = 0
        episodes = []
        assert n_steps or n_episodes, "either n_steps or n_episodes have to be defined"
        while (n_steps and count_steps < n_steps) or (n_episodes and count_episodes < n_episodes):
            episode_infos, resets, obsvs, rewards, dones, infos = env.step(actions)

            episodes += self._reset(episode_infos, resets, experiences)

            for episode, action, reset, *data in zip(self.active_episodes, self.actions, resets, obsvs, rewards, dones, infos):
                if not reset:
                    episode.push(action, *data)

            actions = self._step(self.active_episodes, save_pred=save_pred)

            if n_steps:
                sys.stdout.write(f"\r--- Rolling out {count_steps + 1} / {n_steps} steps")
            else:
                sys.stdout.write(f"\r--- Rolling out {count_episodes + 1} / {n_episodes} episodes")
            sys.stdout.flush()

            for episode, done, info in zip(self.active_episodes, dones, infos):
                if done:
                    if episodes.index(episode) < n_episodes:
                        stats.add_all(self.handler.stats(info))
                        count_episodes += 1

                        if save_pred:
                            episode.save()

            count_steps += len(episode_infos)
        sys.stdout.write("\r")
        sys.stdout.flush()

        for episode in episodes[n_episodes:]:
            episode.delete()

        experiences.save()
        return stats.means(), time.time() - start_time

    def rollouts_(self, env: Env, n_episodes: int = None, train: bool = True,
                  experiences: ExperienceManager = None, save_pred=False):
        start_time = time.time()
        stats = Stats()
        if experiences is None: experiences = self.experiences
        for i_episode in range(n_episodes):
            sys.stdout.write(f"\r--- Rolling out {i_episode + 1} / {n_episodes} episodes")
            sys.stdout.flush()
            stats = self.rollout(env, train=train, experiences=experiences, stats=stats, save_pred=save_pred)
        sys.stdout.write("\r")
        experiences.save()
        sys.stdout.flush()
        return stats.means(), time.time() - start_time

    def rollout(self, env: Env, train: bool = True, experiences: ExperienceManager = None, stats: Stats = None, manual=False, save_pred=False):
        if stats is None: stats = Stats()
        self.training = train
        episode_info, obsv, _ = env.reset()
        if experiences is not None: self.experiences = experiences
        episode = self.reset((episode_info, obsv))[0]
        while True:
            action = self.step(episode, save_pred=save_pred)
            if manual:
                print(f"Type in the action index. The action to choose from are {self.handler.meta.actions}")
                action = self.handler.meta.actions[int(input())]
            obsv, reward, done, info = env.step(action)
            episode.push(action, obsv, reward, done, info)
            if done:
                break
        if save_pred:
            episode.save()
        stats.add_all(self.handler.stats(info))
        return stats

    def policy(self, histories, new_episodes: List[bool], carrier: dict = None):
        if carrier is None: carrier = {}
        outputs = self.trainer.predict(histories, is_train=False, inference=True, new_episodes=new_episodes, **carrier)
        outputs, carrier = self.handler.output_to_carrier(outputs)
        actions = self.trainer.model.action(**{**histories, **outputs}, explore=self.training)
        return self.handler.postproc_actions(actions), outputs, carrier

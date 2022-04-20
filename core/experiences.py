import os
import shutil
import sys
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from core.env import Env
from core.handler import DataHandler
from core.utils.io_utils import JSONIO, DictIO, MultiFileIO
from core.utils.tensor_utils import to_numpy


def concat_data(target_io: Union[DictIO, MultiFileIO], io_key, data: dict):
    target = dict(target_io[io_key])
    if target.keys():
        assert target.keys() == data.keys(), \
            f"Registered {target.keys()} but received {data.keys()}"
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                target[key] = torch.cat([target[key], val.unsqueeze(0)])
            else:
                target[key].append(val)
    else:
        for key, val in data.items():
            target[key] = val.unsqueeze(0) if isinstance(val, torch.Tensor) else [val]
    target_io[io_key] = target


class Episode:
    def __init__(self, manager: 'ExperienceManager', episode_index):
        self.manager = manager
        self.handler = manager.handler
        self.index = episode_index

        self.episode_io = manager.episode_io
        self.obsv_io = manager.obsv_io
        self.trans_io = manager.trans_io
        self.pred_io = manager.pred_io
        self.samples_info = manager.samples_info

    @classmethod
    def create(cls, manager, episode_info, init_obsv, index):
        manager.episode_io[index] = dict(episode_info)
        manager.obsv_io[index] = {}
        manager.trans_io[index] = {}
        manager.pred_io[index] = {}

        obsv_info = manager.handler.preproc_obsv(init_obsv)
        concat_data(manager.obsv_io, index, obsv_info)

        manager.samples_info[index] = 0

        return Episode(manager, index)

    def __len__(self):
        return self.samples_info[self.index]

    def push(self, action, obsv, reward, done, info):
        obsv_info = self.handler.preproc_obsv(obsv)
        concat_data(self.obsv_io, self.index, obsv_info)
        trans_info = {'actions': torch.tensor(self.handler.preproc_action(action), dtype=torch.long),
                      'rewards': torch.tensor(reward, dtype=torch.float),
                      'dones': torch.tensor(done, dtype=torch.bool),
                      **self.handler.preproc_info(info)}
        concat_data(self.trans_io, self.index, trans_info)
        self.samples_info[self.index] += 1

    def push_pred(self, pred):
        concat_data(self.pred_io, self.index, pred)

    def get(self, step=None, inference=False, include_pred=False):
        index = self.index
        if step is None: step = len(self) if inference else len(self) - 1
        assert step >= 0, "Negative step queried"
        frame_len = self.manager.max_train_frames
        if inference:
            start_step = step
        elif frame_len:
            start_step = max(0, step - frame_len)
        else:
            start_step = 0
        episode_info = dict(self.episode_io[index])
        curr_info, past_seq_info, future_seq_info, full_seq_info = \
            self.handler.combine_seq_info(episode_info, self.handler.postproc_obsvs(dict(self.obsv_io[index]), episode_info),
                                          self.handler.postproc_trans(dict(self.trans_io[index])), step, inference, start_step=start_step)
        curr_info = {**episode_info, **curr_info, 'step': torch.tensor(step, dtype=torch.long)}
        result = self.handler.combine_info(curr_info, past_seq_info, future_seq_info, full_seq_info, step, inference)
        if include_pred:
            pred = {k: v[step] for k, v in self.pred_io[index].items()}
            return (*result, pred)
        return result

    def delete(self):
        for io in [self.episode_io, self.obsv_io, self.trans_io, self.pred_io]:
            io.delete(self.index)
        del self.samples_info[self.index]

        self.manager.active_episodes = list(filter(lambda e: e.index != self.index, self.manager.active_episodes))
        self.manager._set_episodes_stats()

    def save(self, save_dir=None):
        epi, past_seq, full_seq, future_seq, pred = self.get(include_pred=True)
        data = {**epi, **past_seq, **full_seq, **future_seq, **pred}
        to_numpy(data)
        if not save_dir:
            save_dir = os.path.join(self.manager.save_dir, "runs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(data, os.path.join(save_dir, f"{self.index}.pt"))


class ExperienceManagerBase(Dataset):
    n_episodes: int


class ExperienceManager(ExperienceManagerBase):
    def __init__(self, handler: DataHandler, save_dir=None, clear: bool = False,
                 episode_in_mem=True, obsv_in_mem=True, trans_in_mem=True, pred_in_mem=True,
                 max_episodes: int = None, cash_size: int = None, max_train_frames=None, **kwargs):
        self.handler = handler
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.cash_size = cash_size
        self.max_train_frames = max_train_frames

        if save_dir:
            if os.path.exists(save_dir):
                if clear:
                    shutil.rmtree(save_dir)
                else:
                    print(f"Loading existing data at {save_dir}")

        self.episode_io = self.get_io(save_dir, "episode", episode_in_mem)
        self.obsv_io = self.get_io(save_dir, "obsv", obsv_in_mem)
        self.trans_io = self.get_io(save_dir, "trans", trans_in_mem)
        self.pred_io = self.get_io(save_dir, "pred", pred_in_mem)
        self.samples_info = JSONIO(os.path.join(save_dir, "samples.json") if save_dir else None)
        self._index = int(sorted(list(self.samples_info.keys()))[-1]) if len(self.samples_info) else 0

        self.active_episodes = self.load_episodes()
        self._set_episodes_stats()

    def load_episodes(self):
        return [Episode(self, i) for i in self.samples_info.keys()]

    def _set_episodes_stats(self):
        self._episodes_len = [len(e) for e in self.active_episodes]
        self._n_samples = np.sum(self._episodes_len, dtype=int)
        self.n_episodes = len(self.active_episodes)

    def get_io(self, save_dir, name, in_mem: bool):
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, name)
            if in_mem: save_path += ".pt"
        return DictIO(path=save_path) if in_mem or not save_path else MultiFileIO(save_path, cash_size=self.cash_size)

    def add_episode(self, episode_info, init_obsv) -> Episode:
        episode = Episode.create(self, episode_info, init_obsv, str(self._index))
        self._index += 1

        if self.max_episodes is not None:
            if len(self.active_episodes) >= self.max_episodes:
                assert len(self.active_episodes) == self.max_episodes
                remove_episode = self.active_episodes.pop(np.random.randint(self.max_episodes))
                remove_episode.delete()
        self.active_episodes.append(episode)
        self._set_episodes_stats()

        return episode

    def __len__(self):
        return self._n_samples

    def __getitem__(self, index):
        _index = index
        for episode, length in zip(self.active_episodes, self._episodes_len):
            if index < length:
                return episode.get(index)
            index -= length
        raise IndexError

    def get_episode(self, i_episode) -> Episode:
        assert 0 <= i_episode < len(self.active_episodes), "episode index out of range"
        return self.active_episodes[i_episode]

    def sample_episodes(self, n=1) -> List[Episode]:
        return [self.active_episodes[i] for i in np.random.permutation(self.active_episodes)[:n]]

    def collect_demos(self, env: Env, n_episodes: int):
        if len(self):
            raise Exception("This experiment has already been run. "
                            "Please rename the save directory of your previous experiment if you want a rerun. "
                            "If you want to overwrite, pass --clear as an argument.")

        print(f"Collecting {n_episodes} demonstrations...")
        for i in range(n_episodes):

            sys.stdout.write(f"\r--- {i + 1} / {n_episodes} episodes")
            sys.stdout.flush()

            episode_info, obsv, opt_actions = env.reset()
            episode = self.add_episode({**episode_info, "is_expert": True}, obsv)

            for action in opt_actions:
                obsv, reward, done, info = env.step(action)
                episode.push(action, obsv, reward, done, info)

        sys.stdout.write("\r\n")
        sys.stdout.flush()

        self.save()

    def save(self):
        if self.save_dir:
            self.episode_io.save()
            self.obsv_io.save()
            self.trans_io.save()
            self.pred_io.save()
            self.samples_info.save()


class MultiExperienceManager(ExperienceManagerBase):
    def __init__(self, managers: List[ExperienceManagerBase] = None, max_managers: int = None):
        self.managers = managers or []
        self.max_managers = max_managers

    def add(self, manager: ExperienceManagerBase):
        if self.max_managers is not None:
            self.managers = self.managers[-(self.max_managers - 1):] if self.max_managers > 1 else []
        self.managers.append(manager)

    def __len__(self):
        return np.sum([len(manager) for manager in self.managers], dtype=int)

    def __getitem__(self, index):
        for manager in self.managers:
            length = len(manager)
            if index < length:
                return manager[index]
            index -= length
        raise IndexError

    @property
    def n_episodes(self):
        return np.sum([manager.n_episodes for manager in self.managers], dtype=int)

    def get_episode(self, index):
        for manager in self.managers:
            length = manager.n_episodes
            if index < length:
                return manager.get_episode(index)
            index -= length
        raise IndexError("episode index out of range")


def get_experience_loader(handler, data=None, is_train=True, batch_size=None, **config):
    dataset = ExperienceManager(handler, save_dir=os.path.join(data, "train" if is_train else "val"), **config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, collate_fn=handler.collate, drop_last=True)
    return loader, dataset

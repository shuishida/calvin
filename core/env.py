from typing import Tuple, List, Any, Optional, Callable

from core.utils.image_utils import tile_images
from core.mdp.meta import MDPMeta


class Env:
    def __init__(self, meta: MDPMeta, max_steps=None):
        self.meta = meta
        self.max_steps = max_steps
        self.count_steps = 0
        self.total_rewards = 0
        self.opt_actions = None
        assert max_steps, "max_steps needs to be set"

    def reset(self) -> Tuple[dict, Any, Optional[List[Any]]]:
        """
        :return: tuple of episode_info (dict), initial observation, and optionally, the optimal set of actions (if available)
        """
        self.count_steps = 0
        self.total_rewards = 0
        episode_info, init_obsv, opt_actions = self._reset()
        self.opt_actions = opt_actions
        return episode_info, init_obsv, opt_actions

    def _reset(self) -> Tuple[dict, Any, Optional[List[Any]]]:
        """
        :return: tuple of episode_info (dict), initial observation, and optionally, the optimal set of actions (if available)
        """
        pass

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        """
        :param action:
        :return: (obsv, reward, done, info)
        """
        obsv, reward, done, info = self._step(action)
        self.total_rewards += reward
        self.count_steps += 1
        info = {**info, 'total_rewards': self.total_rewards, 'total_steps': self.count_steps,
                'opt_steps': len(self.opt_actions)}
        if not done and self.count_steps == self.max_steps:
            done = True
        return obsv, reward, done, info

    def _step(self, action) -> Tuple[Any, float, bool, dict]:
        """
        :param action:
        :return: (obsv, reward, done, info)
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """
        :param mode: 'human' or 'rgb_array'
        :return: If 'rgb_array', return rgb numpy array. If 'human', show display.
        """
        pass

    def close(self) -> None:
        pass


class RemoteEnv:
    def __init__(self, env):
        self.env = env
        self.episode_info = None
        self.done = True

    def step(self, action) -> Tuple[Tuple[dict, dict], bool, Any, float, bool, Any]:
        if self.done or action is None:
            episode_info, obsv, _ = self.env.reset()
            self.episode_info = (episode_info, obsv)
            reward, self.done, info, reset = 0, False, None, True
        else:
            obsv, reward, self.done, info = self.env.step(action)
            reset = False
        return self.episode_info, reset, obsv, reward, self.done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self) -> None:
        self.env.close()


class VecEnv:
    def __init__(self, env_fns: List[Callable[[], Env]]):
        self.envs = [RemoteEnv(env_fn()) for env_fn in env_fns]
        self.size = len(self.envs)

    def step(self, actions: List[Any] = None) -> Tuple[List[Tuple[dict, Any]], List[bool], List[Any], List[float], List[bool], List[Any]]:
        if actions is None: actions = [None] * self.size
        assert len(actions) == self.size, "size of action must equal the number of environments"
        results = [env.step(action) for env, action in zip(self.envs, actions)]

        return list(map(list, zip(*results)))

    def render(self, mode='human'):
        """
        :param mode: 'human' or 'rgb_array'
        :return: If 'rgb_array', return rgb numpy array. If 'human', show display.
        """
        imgs = [env.render("rgb_array") for env in self.envs]

        if imgs[0] is None: return

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == "human":
            import cv2  # pytype:disable=import-error

            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")

    def close(self) -> None:
        for env in self.envs:
            env.close()


class MockVecEnv:
    def __init__(self, env: Env):
        self.env = env
        self.episode_info = None
        self.done = True

    @property
    def size(self):
        return 1

    def step(self, actions: List[Any] = None) -> Tuple[List[Tuple[dict, Any]], List[bool], List[Any], List[float], List[bool], List[Any]]:
        assert actions is None or len(actions) == self.size, "size of action must equal the number of environments"

        if self.done or actions is None:
            episode_info, obsv, _ = self.env.reset()
            self.episode_info = (episode_info, obsv)
            reset, reward, self.done, info = True, 0, False, None
        else:
            obsv, reward, self.done, info = self.env.step(actions[0])
            reset = False
        return [self.episode_info], [reset], [obsv], [reward], [self.done], [info]

    def close(self) -> None:
        self.env.close()
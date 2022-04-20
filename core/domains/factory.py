from dataclasses import dataclass
from typing import Callable, Type

from core.env import Env
from core.handler import DataHandler
from core.mdp.meta import EnvMeta


@dataclass
class Factory:
    meta: Callable[..., EnvMeta]
    env: Callable[..., Env]
    model_config: Callable[..., dict]
    handler: Type[DataHandler]


def get_factory(domain) -> Factory:
    if domain == "grid":
        from core.domains.gridworld.factory import init_grid_meta, init_grid_env, get_grid_env_model_config
        from core.domains.gridworld.handler import GridDataHandler
        return Factory(meta=init_grid_meta, env=init_grid_env,
                       model_config=get_grid_env_model_config, handler=GridDataHandler)
    elif domain == "miniworld":
        from core.domains.miniworld.factory import init_miniworld_meta, init_miniworld_env, get_miniworld_model_config
        from core.domains.miniworld.handler import MiniWorldDataHandler
        return Factory(meta=init_miniworld_meta, env=init_miniworld_env,
                       model_config=get_miniworld_model_config, handler=MiniWorldDataHandler)
    elif domain == "avd":
        from core.domains.avd.factory import init_avd_meta, init_avd_env, get_avd_model_config
        from core.domains.avd.handler import AVDDataHandler
        return Factory(meta=init_avd_meta, env=init_avd_env,
                       model_config=get_avd_model_config, handler=AVDDataHandler)
    else:
        raise Exception(f"domain {domain} not registered")

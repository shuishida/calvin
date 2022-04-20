import sys

sys.path.append(".")
from core.domains.miniworld.factory import init_miniworld_meta, init_miniworld_env
from core.domains.miniworld.handler import MiniWorldDataHandler


import matplotlib.pyplot as plt


env_config = {
    'size': 3,
    'map_bbox': (0, 0, 10, 10),
    'map_res': (30, 30),
    'ori_res': 8,
    'costmap_margin': 5,
    'min_traj_len': 0,
    'max_steps': 500,
    'sample_free': 4
}

meta = init_miniworld_meta(**env_config)
env = init_miniworld_env(meta, **env_config)
handler = MiniWorldDataHandler(meta)

episode_info, obsv, opt_actions = env.reset()
print(episode_info.keys(), obsv.keys(), opt_actions)

for action in opt_actions:
    obsv, reward, done, info = env.step(action)
    print(obsv.keys(), obsv['poses'], action, reward, done, episode_info.keys())
    plt.imshow(obsv['top_view'])
    plt.show()

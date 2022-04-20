import sys

sys.path.append(".")

# This import is necessary
import core.domains.miniworld
import pyglet
pyglet.options['headless'] = False

from core.domains.miniworld.factory import init_miniworld_meta, init_miniworld_env
from core.domains.miniworld.handler import MiniWorldDataHandler
from einops import rearrange
import pptk
import torch
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

episode_info, _, opt_actions = env.reset()

xyzs = []
rgbs = []

for i in range(10):
    action = int(input())
    obsv, reward, done, info = env.step(action)
    print(env.env.agent.pos, env.env.agent.dir)
    xyz = rearrange(obsv['surf_xyz'], "f h w -> (h w) f")
    xyzs.append(xyz)
    rgb = rearrange(obsv['rgb'], "f h w -> (h w) f")
    rgbs.append(rgb)
    plt.imshow(obsv['top_view'])
    plt.show()
    v = pptk.viewer(torch.cat(xyzs))
    v.attributes(torch.cat(rgbs) / 255)
    v.set(point_size=0.03)

"""
python core/domains/miniworld/examples/manual.py
"""

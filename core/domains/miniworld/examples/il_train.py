import sys

from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append(".")

from core.env import VecEnv

from core.models.calvin.calvin_conv3d import CALVINConv3d
from core.domains.miniworld.factory import init_miniworld_meta, init_miniworld_env
from core.domains.miniworld.handler import MiniWorldDataHandler
from core.agent import MemoryAgent
from core.experiences import ExperienceManager
from core.trainer import Trainer


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

expert_demos = ExperienceManager(handler, save_dir="data/rl/expert", clear=True)
expert_demos.collect_demos(env, n_episodes=400)
model = CALVINConv3d(l_i=3, l_h=150, k_sz=3, ori_res=8, action_set=meta.actions, k=40, discount=0.25)

optimizer = Adam(model.parameters(), lr=0.005, eps=1e-6)

trainer = Trainer(model, optimizer)

agent = MemoryAgent(handler, trainer)

loader = DataLoader(expert_demos, batch_size=128, shuffle=True, collate_fn=handler.collate, drop_last=True)

vecenv = VecEnv([lambda: init_miniworld_env(meta, **env_config) for _ in range(4)])

for i in range(10):
    stats_train, dur = agent.trainer.fit_epoch(loader, is_train=True)
    stats_nav, _ = agent.rollouts(vecenv, n_episodes=100, train=False,
                               experiences=ExperienceManager(handler, save_dir=f"data/rl/epoch_{i}", clear=True))
    print("Epoch:", i, {**stats_train, **stats_nav})


import sys

from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append(".")

from core.env import VecEnv

from core.models.calvin.calvin_conv2d import CALVINConv2d
from core.agent import MemoryAgent
from core.domains.gridworld.handler import GridDataHandler
from core.experiences import ExperienceManager
from core.trainer import Trainer
from core.domains.gridworld.factory import init_grid_meta, init_grid_env


# episode_info, obsv, opt_actions = env.reset()
# print(episode_info.keys(), obsv.keys(), opt_actions)
#
# for action in opt_actions:
#     obsv, reward, done, info = env.step(action)
#     print(obsv.keys(), reward, done)


env_config = {
    'size': (15, 15),
    'four_way': False,
    'map': 'maze',
    'ego': False,
    'view_range': 2,
    'min_traj_len': 15,
    'max_steps': 300,
    'allow_backward': False
}


meta = init_grid_meta(**env_config)
env = init_grid_env(meta, **env_config)
handler = GridDataHandler(meta)

expert_demos = ExperienceManager(handler, save_dir="data/rl/expert", clear=True)
expert_demos.collect_demos(env, n_episodes=4000)
model = CALVINConv2d(l_i=3, l_h=150, k_sz=3, action_set=meta.actions, k=40, discount=0.25)
# model = SimpleModel()
optimizer = Adam(model.parameters(), lr=0.005, eps=1e-6)

trainer = Trainer(model, optimizer)

agent = MemoryAgent(handler, trainer)

loader = DataLoader(expert_demos, batch_size=128, shuffle=True, collate_fn=handler.collate, drop_last=True)

vecenv = VecEnv([lambda: init_grid_env(meta, **env_config) for _ in range(4)])

for i in range(10):
    stats_train, dur = agent.trainer.fit_epoch(loader, is_train=True)
    stats_nav, _ = agent.rollouts(vecenv, n_episodes=100, train=False,
                               experiences=ExperienceManager(handler, save_dir=f"data/rl/epoch_{i}", clear=True))
    print("Epoch:", i, {**stats_train, **stats_nav})


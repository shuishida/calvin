import json
import os
import sys
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.utils.utils import print_items

sys.path.append(".")

from core.env import VecEnv
from core.utils.logger import MetaLogger
from core.utils.trainer_utils import setup_agent
from core.experiences import ExperienceManager, MultiExperienceManager


class AgentTrainer:
    def __init__(self, data=None, name=None, batch_size=None, max_train_frames=None, n_workers=None, n_envs=None,
                 cash_size=None,
                 epochs_per_iter=None, **config):
        env_config, meta, self.handler, self.trainer, self.agent, init_env = setup_agent(data=data, name=name, **config)

        self._init_vecenv(env_config, meta, init_env, n_envs)
        self.env_config = env_config
        self.data = data
        self.batch_size = batch_size
        self.max_train_frames = max_train_frames
        self.n_workers = n_workers
        self.cash_size = cash_size
        self.epochs_per_iter = epochs_per_iter
        self.writer = SummaryWriter()

        self.save_dir = os.path.join(data, "models",
                                     self.get_save_name(name if name else repr(self.trainer.model),
                                                        **config) + "_" + datetime.now().strftime("%m%d_%H%M%S_%f"))

        os.makedirs(self.save_dir)

        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(self.trainer.config, f)

        self.best_val_so_far = None

    def _init_vecenv(self, env_config, meta, init_env, n_envs):
        self.vecenv_train = VecEnv(
            [lambda: init_env(meta, i_env=i_env, split="train", **env_config) for i_env in range(n_envs)])
        self.vecenv_val = VecEnv(
            [lambda: init_env(meta, i_env=i_env, split="val", **env_config) for i_env in range(n_envs)])

    def get_save_name(self, name, **config):
        raise NotImplementedError

    def get_experiences(self, save_dir=None):
        return ExperienceManager(self.handler, save_dir=save_dir, cash_size=self.cash_size,
                                 max_train_frames=self.max_train_frames, **self.env_config)

    def get_data_loader(self, dataset, is_train=True):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=is_train,
                          num_workers=self.n_workers, drop_last=True, collate_fn=self.handler.collate)

    def train(self, epochs=None, buffer_size=None, n_rollouts=0, n_evals=100, save_pred=False, n_eval_train=None, **kwargs):
        print(os.path.join(os.getcwd(), self.save_dir))

        agent = self.agent

        meta_manager = MultiExperienceManager()

        if buffer_size > 0:
            manager = MultiExperienceManager(max_managers=buffer_size)
            meta_manager.add(manager)
        else:
            manager = None

        expert_demos = self.get_experiences(os.path.join(self.data, "train"))
        meta_manager.add(expert_demos)

        val_dataset = self.get_experiences(os.path.join(self.data, "val"))
        val_loader = self.get_data_loader(val_dataset, is_train=False) if len(val_dataset) else None

        self.print_header()

        with MetaLogger(self.save_dir, self.trainer.config) as logger:
            for epoch in range(epochs):
                # epoch_save_dir = os.path.join(self.save_dir, f"epoch_{epoch:03d}")
                experiences = self.get_experiences()

                if n_rollouts:
                    rollout_stats, rollout_dur = agent.rollouts(self.vecenv_train, n_episodes=n_rollouts, train=True,
                                                                experiences=experiences)
                else:
                    rollout_stats, rollout_dur = {}, 0

                if buffer_size > 0: manager.add(experiences)

                train_loader = self.get_data_loader(meta_manager)
                train_stats, train_dur = agent.trainer.fit_epoch(train_loader, is_train=True,
                                                                 save_path=os.path.join(self.save_dir, "train.pt"),
                                                                 epochs=self.epochs_per_iter,
                                                                 logger=logger)

                val_stats, val_dur = agent.trainer.fit_epoch(val_loader, is_train=False,
                                                             save_path=os.path.join(self.save_dir, "val.pt"),
                                                             logger=logger) if val_loader else ({}, 0)

                # epoch_save_dir = os.path.join(self.save_dir, f"epoch_{epoch:03d}_eval")
                if n_eval_train:
                    experiences = self.get_experiences()
                    eval_train_stats, eval_train_dur = agent.rollouts(self.vecenv_train, n_episodes=n_eval_train, train=False,
                                                                      experiences=experiences, save_pred=False)
                else:
                    eval_train_stats, eval_train_dur = {}, 0

                if n_evals:
                    experiences = self.get_experiences()
                    eval_stats, eval_dur = agent.rollouts(self.vecenv_val, n_episodes=n_evals, train=False,
                                                          experiences=experiences, save_pred=save_pred)
                else:
                    eval_stats, eval_dur = {}, 0

                data = {'epoch': epoch,
                        **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"val_{k}": v for k, v in val_stats.items()},
                        **{f"rollout_{k}": v for k, v in rollout_stats.items()},
                        **{f"eval_train_{k}": v for k, v in eval_train_stats.items()},
                        **{f"eval_{k}": v for k, v in eval_stats.items()},
                        'train_dur': train_dur, 'rollout_dur': rollout_dur, 'eval_dur': eval_dur
                        }

                sys.stdout.write("\r")
                self.print_stats(**data)
                self.save_best(data)

                logger.append(data, epoch)

                self.trainer.save_checkpoint(os.path.join(self.save_dir, f"epoch_{epoch:03d}", "checkpoint.pt"))

    def save_best(self, stats):
        metric = "val_loss"
        if self.best_val_so_far is None or stats[metric] < self.best_val_so_far:
            self.best_val_so_far = stats[metric]

            with open(os.path.join(self.save_dir, "best_stats.json"), "w") as f:
                json.dump(stats, f)

    def print_header(self):
        print_items("epoch", "train loss", "val loss", "rollout acc.", "eval train", "eval acc.", "avg. reward",
                    "duration")

    def print_stats(self, epoch, train_loss=None, val_loss=None, rollout_acc=None, eval_train_acc=None, eval_acc=None,
                    eval_reward=None, train_dur=0, val_dur=0, rollout_dur=0, **kwargs):
        print_items(epoch, train_loss, val_loss, rollout_acc, eval_train_acc, eval_acc, eval_reward,
                    train_dur + val_dur + rollout_dur)


def add_rollout_core_args(parser):
    parser.add_argument("--n_trials", "-ntr", type=int, help="number of trials", default=100)
    parser.add_argument("--n_envs", "-env", type=int, help="number of envs", default=4)
    parser.add_argument('--softmax', '-sm', action='store_true', default=False,
                        help="take softmax in the action policy to introduce stochasticity")
    return parser


def add_train_args(parser):
    parser.add_argument("--data", help="path to data directory")
    parser.add_argument("--name", help="name of save directory")
    parser.add_argument("--device", "-d", default="cuda", help="device (cuda, cuda:0, cpu, etc.)")
    parser.add_argument("--model", help="model class or path to model")
    parser.add_argument("--checkpoint", '-cp', help="checkpoint")
    parser.add_argument("--batch_size", "-bs", default=128, type=int, help="batch size")
    parser.add_argument("--buffer_size", "-bf", default=1, type=int, help="buffer size")
    parser.add_argument("--epochs_per_iter", "-epi", default=1, type=int, help="epochs per iteration")
    parser.add_argument("--n_workers", "-nw", default=0, type=int, help="number of workers")
    parser.add_argument("--cash_size", "-cs", default=1000, type=int, help="cash size")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--optim', default="adam", help='Optimizer to use (adam, rms)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--clip', type=float, default=0.1, help='Gradient clipping')
    parser.add_argument('--save_pred', action="store_true", help="Save prediction")
    parser.add_argument("--n_rollouts", "-nro", type=int, help="number of rollouts (0 if imitation learning)",
                        default=0)
    parser.add_argument("--n_evals", "-nev", type=int, help="number of evaluation runs", default=100)
    parser.add_argument("--n_eval_train", "-nevt", type=int, help="number of evaluation runs", default=0)
    parser.add_argument("--n_envs", "-env", type=int, help="number of envs", default=8)
    parser.add_argument("--softmax", "-sm", type=float, default=0, help="boltzmann action selection temperature")
    parser.add_argument("--max_train_frames", "-frames", default=None, type=int,
                        help="Maximum number of frames at training time")
    parser.add_argument('--require_returns', '-rr', action='store_true', default=False,
                        help="require returns")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    return parser

import argparse
import sys

sys.path.append(".")

from core.models.projection.point_cloud_vin_base import add_pcn_train_args
from core.models.vin.vin_base import add_vin_args
from core.agent_trainer import AgentTrainer, add_train_args


class AVDEnvTrainer(AgentTrainer):
    def get_save_name(self, name, optim=None, lr=None, clip=None, k_sz=None, **kwargs):
        return f"{name}_{k_sz}_{optim}_{lr}_{clip}"


def add_avd_train_args(parser):
    add_train_args(parser)
    add_pcn_train_args(parser)
    parser.add_argument("--v_bbox", help="vertical bounding box",
                        type=float, nargs=2, default=(-4, 4))
    parser.add_argument("--v_res", help="vertical resolution",
                        type=int, default=10)
    parser.add_argument('--min_traj_len', '-trajlen', type=int, help="minimum threshold of trajectory length")
    parser.add_argument('--target_size_ratio', '-tsr', default=0.2, type=float, help="target size ratio")
    return parser


def main():
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    add_avd_train_args(parser)
    add_vin_args(parser)
    config = vars(parser.parse_args())

    if config['min_traj_len'] is None: del config['min_traj_len']

    trainer = AVDEnvTrainer(**config)
    trainer.train(**config)


if __name__ == "__main__":
    main()

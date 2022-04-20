import argparse
import sys


sys.path.append(".")

from core.models.vin.vin_base import add_vin_args
from core.agent_trainer import AgentTrainer, add_train_args


class GridEnvTrainer(AgentTrainer):
    def get_save_name(self, name, optim=None, lr=None, clip=None, sparse=None, dense=None,
                      discount=None, target_known=None, **kwargs):
        return f"{name}_{optim}_{lr}_{clip}{'_sp' if sparse else ''}{'_dns' if dense else ''}_{discount}" \
               f"{'_known' if target_known else ''}"


def main():
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    add_train_args(parser)
    add_vin_args(parser)
    config = vars(parser.parse_args())

    trainer = GridEnvTrainer(**config)
    trainer.train(**config)


if __name__ == "__main__":
    main()

import json
import os

import torch
from torch.optim import Adam, RMSprop

from core.agent import MemoryAgent
from core.domains.factory import get_factory
from core.models.calvin.calvin_conv2d import CALVINConv2d, CALVINPosNav
from core.models.calvin.calvin_conv3d import CALVINConv3d, CALVINPoseNav
from core.models.gppn.gppn import GPPN, GPPNNav, GPPNPose, GPPNPoseNav
from core.models.vin.vin import VIN, VINPosNav
from core.models.vin.vin_pose import VINPose, VINPoseNav
from core.trainer import Trainer


def get_model_class_from_name(name):
    available = [CALVINConv2d, CALVINConv3d, CALVINPosNav, CALVINPoseNav,
                 VIN, VINPose, VINPosNav, VINPoseNav,
                 GPPN, GPPNNav, GPPNPose, GPPNPoseNav]
    model_dict = {model.__name__: model for model in available}
    return model_dict[name]


def create_model(model=None, device=None, optim=None, lr=None, **config):
    assert model, "No model name defined"
    Model = get_model_class_from_name(model)
    # Instantiate a VIN model
    model = Model(**config, device=device).to(device)
    if list(model.parameters()):
        # Optimizer
        if optim == "adam":
            optimizer = Adam(model.parameters(), lr=lr, eps=1e-6)
        elif optim == "rms":
            optimizer = RMSprop(model.parameters(), lr=lr, eps=1e-6)
        else:
            raise Exception(f"Unknown optimizer: {optim}")
    else:
        optimizer = None
    return model, optimizer


def setup_trainer(data=None, device="cuda", checkpoint=None, clip=None, **config):
    if not torch.cuda.is_available(): device = "cpu"

    with open(os.path.join(data, "env_config.json"), "r") as f:
        env_config = json.load(f)
        if "seed" in env_config:
            del env_config['seed']

    if checkpoint:
        with open(checkpoint + ".json", "r") as f:
            config = json.load(f)

    env_config = {**env_config, **config}

    domain = env_config['domain']
    factory = get_factory(domain)
    model_config = factory.model_config(dict(config), **env_config)
    model_config['device'] = device
    model, optimizer = create_model(**model_config)

    meta = factory.meta(**env_config)
    handler = factory.handler(meta, **env_config)
    init_env = factory.env

    trainer = Trainer(model, optimizer, config, checkpoint=checkpoint, clip=clip)

    return env_config, meta, handler, trainer, init_env


def setup_agent(*args, **kwargs):
    env_config, meta, handler, trainer, init_env = setup_trainer(*args, **kwargs)
    agent = MemoryAgent(handler, trainer)
    return env_config, meta, handler, trainer, agent, init_env


def run_epochs(trainer, train_loader, val_loader, epochs, save_dir):
    for epoch in range(epochs):
        stats_train, dur = trainer.fit_epoch(train_loader, is_train=True, save_path=os.path.join(save_dir, "train.pt"))
        print("Epoch:", epoch, stats_train)

        stats_val, dur = trainer.fit_epoch(val_loader, is_train=False, save_path=os.path.join(save_dir, "val.pt"))
        print("Epoch:", epoch, stats_val)

        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch:03}")
        os.makedirs(epoch_save_dir)
        trainer.save_checkpoint(os.path.join(epoch_save_dir, "checkpoint.pt"))

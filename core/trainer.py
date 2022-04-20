import json
import os
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from core.model import Model
from core.utils.logger import MetaLogger
from core.utils.tensor_utils import to_numpy
from core.utils.utils import Stats


class Trainer:
    def __init__(self, model: Model, optimizer, config=None,
                 checkpoint: str = None, clip: float = None, clear: bool = False, save_interval: int = 300, **kwargs):
        self.config = config
        self.device = model.device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.clip = clip
        self.clear = clear
        self.save_interval = save_interval
        self.start_time = time.time()

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def save_checkpoint(self, checkpoint_path, **data):
        dirpath = os.path.dirname(checkpoint_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        torch.save({
            'arch': type(self.model).__name__,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler': self.scheduler.state_dict(),
            **data
        }, checkpoint_path)
        with open(checkpoint_path+".json", "w") as f:
            json.dump(self.config, f)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
        with open(checkpoint_path+".json", "r") as f:
            self.config = json.load(f)

    def predict(self, inputs: dict, is_train: bool, **settings):
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        self.model.train() if is_train else self.model.eval()
        with torch.enable_grad() if is_train else torch.no_grad():
            if hasattr(self.model, "preprocess"): self.model.preprocess(inputs, **settings)
            outputs = self.model(**inputs, **settings)

        return outputs

    def forward_pass(self, inputs: dict, is_train: bool):
        outputs = self.predict(inputs, is_train)

        loss_batch, loss_outputs = self.model.loss(**{**inputs, **outputs})
        stats = {'loss': loss_batch.item()}
        if hasattr(self.model, "metrics"):
            stats = {**stats, **self.model.metrics(**{**inputs, **outputs, **loss_outputs})}

        return {**outputs, **loss_outputs}, loss_batch, stats

    def fit_epoch(self, loader: DataLoader, is_train=True, epochs=1, save_path=None, logger: MetaLogger=None, **settings):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        self.model.train() if is_train else self.model.eval()

        with torch.enable_grad() if is_train else torch.no_grad():
            sum_acc, num_batches, i_batch = 0.0, len(loader) * epochs, 0
            stats_collector = Stats()
            start_time = time.time()

            for _ in range(epochs):
                last_saved = None
                for inputs in loader:  # Loop over batches of data
                    inputs = {**inputs, **settings}
                    outputs, loss_batch, stats = self.forward_pass(inputs, is_train)
                    stats_collector.add_all(stats)

                    if is_train:
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # backward pass
                        loss_batch.backward()
                        if self.clip:
                            # clip gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    if is_train:
                        self.optimizer.step()

                    if save_path:
                        needs_updating = last_saved is None or time.time() > last_saved + self.save_interval
                        if needs_updating:
                            torch.save(to_numpy({
                                **inputs, **outputs, 'saved_time': datetime.now().strftime('%m%d_%H%M%S_%f')
                            }), save_path)
                            last_saved = time.time()

                    i_batch += 1
                    sys.stdout.write(f"\r--- {i_batch} / {num_batches} batches; avg. loss: {loss_batch}")
                    sys.stdout.flush()

            sys.stdout.write("\r")
            sys.stdout.flush()

            time_duration = time.time() - start_time

            # if is_train: self.scheduler.step()

        return stats_collector.means(), time_duration

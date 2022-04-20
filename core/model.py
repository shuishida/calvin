from typing import Tuple

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, device="cuda", **kwargs):
        super(Model, self).__init__()
        self.device = device if torch.cuda.is_available() else "cpu"

    def forward(self, **data) -> dict:
        raise NotImplementedError

    def loss(self, **data) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def metrics(self, **data) -> dict:
        return {}

    def action(self, **data):
        raise NotImplementedError

    def preprocess(self, data: dict, **kwargs):
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

# from pytorch_memlab import profile, set_target_gpu
from torch.nn import Parameter, Module

from models.utils.torch_bin_reduce.bin_reduce import BinReduce
from math import ceil


class ConvDeconv(nn.Module):
    def __init__(self, map_bbox=None, map_res=None, pcn_i=None, pcn_f=None, big_maze=False, device=None, **kwargs):
        super(ConvDeconv, self).__init__()
        self.map_bbox = map_bbox        # (h1, w1, h2, w2)
        self.map_res = map_res          # (map_x, map_z)
        self.big_size = big_maze
        self.device = device

        sigma = torch.tensor(1.0).to(self.device)
        self.sigma = Parameter(sigma, requires_grad=True)
        Module.register_parameter(self, "sigma", self.sigma)
        mu = torch.tensor(0.0).to(self.device)
        self.mu = Parameter(mu, requires_grad=True)
        Module.register_parameter(self, "mu", self.mu)

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(pcn_i),
            nn.Conv2d(in_channels=pcn_i, out_channels=64, dilation=2,
                      kernel_size=3, stride=1, padding=2, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, dilation=2,
                      kernel_size=3, stride=1, padding=2, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 7, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 128 * 4 * 4),
            nn.Dropout(0.2),
            nn.ReLU()
        ) if not big_maze else nn.Sequential(
            nn.Linear(128 * 5 * 7, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 128 * 5 * 5),
            nn.Dropout(0.2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, pcn_f, 2, stride=2, padding=1)
        ) if not big_maze else nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, pcn_f, 2, stride=2)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}"

    # @profile
    def forward(self, *, agent_view=None, view_xyz=None, index=None, **kwargs):
        """
        N = sum_{batch}{traj_len * orientation}, H = image height, W = image width, 3 = (X, Y, Z)
        :param rgb:            (N, 3, H, W) RGB for each surface point
        :param surf_xyz:        (N, 3, H, W) XYZ for each surface point
        :param free_xyz:        (N, 3, H, W, k) XYZ for each sampled point
        :param valid_points:    (N, H, W) boolean for each point
        :param index:           (N, H, W) batch element index for each point
        :return: feature_map:   (B, F, X, Y)
        """
        rgb = rearrange(agent_view, "n h w f -> n f h w").float()
        xyz = rearrange(view_xyz, "n h w f -> n f h w").float()
        xyz = (xyz - self.mu) / F.softplus(self.sigma)
        N, _, H, W = rgb.size()
        x = torch.cat([rgb, xyz], dim=1)
        x = rearrange(self.encoder(x), "n f h w -> n (f h w)")
        x = self.fc(x)
        x = x.view(rgb.size(0), 128, 4, 4) if not getattr(self, "big_size", False) else x.view(rgb.size(0), 128, 5, 5)
        x = self.decoder(x)

        return x, torch.ones_like(x)


class ConvDeconvVINBase:
    def __init__(self, **config):
        super().__init__(**config)
        self.point_conv_net = self.get_point_conv_model(**config)

    def get_point_conv_model(self, **config):
        return ConvDeconv(**config)

    def __repr__(self):
        return f"{super().__repr__()}_{repr(self.point_conv_net)}"

    # @profile
    def forward(self, *, k=None, prev_v=None, **kwargs):
        feature_map, features_exist = self.point_conv_net(**kwargs)
        outputs = super().forward(feature_map, k=k, prev_v=prev_v)
        return {
            **outputs,
            'features_exist': features_exist
        }

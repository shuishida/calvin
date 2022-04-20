# from pytorch_memlab import profile, set_target_gpu
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch_scatter import scatter

from core.models.detector.resnet.resnet_feature_extractor import PretrainedResNetShort


def selector_to_index(selector, output_sizes):
    B, V, H, W = output_sizes
    return selector[:, 0] * V * H * W + selector[:, 1] * H * W + selector[:, 2] * W + selector[:, 3]


def get_counts(index, output_sizes):
    feature_counts = torch.bincount(index, minlength=np.prod(output_sizes))
    return feature_counts.view(*output_sizes).detach()


def bin_reduce(data, selector, output_sizes):
    index = selector_to_index(selector, output_sizes)
    output = scatter(data, index, dim=0, dim_size=np.prod(output_sizes), reduce="sum")
    return output.view(*output_sizes, data.size(-1)), get_counts(index, output_sizes)


class PointCloudTo2D(nn.Module):
    def __init__(self, map_bbox=None, map_res=None, pcn_h=None, pcn_i=None, pcn_f=None, v_bbox=None, v_res=None,
                 dropout=None,
                 xyz_to_h=None, xyz_to_w=None, pcn_sample_ratio=None, noise_ratio=0.0, use_batch_norm=False,
                 use_resnet=False, use_embeddings=None,
                 device=None, dot_channels=None, **kwargs):
        super(PointCloudTo2D, self).__init__()
        self.map_bbox = map_bbox  # (h1, w1, h2, w2)
        self.map_res = map_res  # (map_x, map_z)
        self.h = h = pcn_h  # size of hidden layers
        self.f = f = pcn_f  # size of hidden layers
        self.xyz_to_h = xyz_to_h  # xyz dim corresponding to h dim
        self.xyz_to_w = xyz_to_w  # xyz dim corresponding to w dim
        axes = {0, 1, 2}
        axes.remove(xyz_to_h)
        axes.remove(xyz_to_w)
        self.xyz_to_v = axes.pop()
        self.pcn_sample_ratio = pcn_sample_ratio  # number of output features

        self.v_bbox = v_bbox
        self.v_res = v_res

        if use_resnet:
            print("Using ResNet18...")
            self.resnet = PretrainedResNetShort(freeze=True, device=device, cutoff_layers=5)
            self.resnet.eval()

            print(self.resnet.model)

            self.rgb_conv_net = nn.Sequential(
                *((nn.GroupNorm(8, 64),) if use_batch_norm else ()),
                nn.Conv2d(in_channels=64, out_channels=h,
                          kernel_size=(3, 3), dilation=(2, 2), stride=(1, 1), padding=(2, 2), bias=False),
                nn.Dropout(dropout),
                nn.ReLU(),
                *((nn.GroupNorm(8, h),) if use_batch_norm else ()),
                nn.Conv2d(in_channels=h, out_channels=h, dilation=(2, 2),
                          kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        elif use_embeddings:
            self.resnet = None

            self.rgb_conv_net = nn.Sequential(
                *((nn.GroupNorm(8, pcn_i),) if use_batch_norm else ()),
                nn.Conv2d(in_channels=pcn_i, out_channels=h, dilation=(2, 2),
                          kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
                nn.Dropout(dropout),
                nn.ReLU(),
                *((nn.GroupNorm(8, h),) if use_batch_norm else ()),
                nn.Conv2d(in_channels=h, out_channels=h, dilation=(2, 2),
                          kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
                nn.Dropout(dropout),
                nn.ReLU()
            )
        else:
            self.resnet = None

            self.rgb_conv_net = nn.Sequential(
                *((nn.GroupNorm(8, pcn_i),) if use_batch_norm else ()),
                nn.Conv2d(in_channels=pcn_i, out_channels=h, dilation=(2, 2),
                          kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                *((nn.GroupNorm(8, h),) if use_batch_norm else ()),
                nn.Conv2d(in_channels=h, out_channels=h, dilation=(2, 2),
                          kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=False),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.scale = 4
        self.use_batch_norm = use_batch_norm
        self.use_embeddings = use_embeddings

        self.noise_ratio = noise_ratio

        self.pre_dot_layer = nn.Conv2d(in_channels=pcn_i, out_channels=dot_channels * h,
                                       kernel_size=(5, 5), stride=(1, 1), bias=True)
        self.post_dot_layer = nn.Conv2d(in_channels=dot_channels, out_channels=h,
                                        kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.device = device

    def __repr__(self):
        return f"{self.__class__.__name__}_h_{self.h}{'_bn' if self.use_batch_norm else ''}"

    # @profile
    def forward(self, *, rgb=None, emb=None, surf_xyz=None, index=None, valid_points=None, free_xyz=None,
                target_emb=None, batch_size=None, **kwargs):
        """
        N = sum_{batch}{traj_len * orientation}, H = image height, W = image width, 3 = (X, Y, Z)
        :param rgb:            (N, 3, H, W) RGB for each surface point
        :param surf_xyz:        (N, 3, H, W) XYZ for each surface point
        :param free_xyz:        (N, H, W, k, 3) XYZ for each surface point
        :param valid_points:    (N, H, W) boolean for each point
        :param index:           (N, H, W) batch element index for each point
        :return: feature_map:   (B, F, X, Y)
        """
        scale = self.scale
        # N, H, W, K, _ = free_xyz.size()
        N, _, H, W = surf_xyz.size()
        if self.use_embeddings:
            surf_rgb = rgb
            surf_features = self.rgb_conv_net(emb)
            _index = index.view(-1, 1, 1).expand(-1, H, W)
        else:
            surf_rgb = F.avg_pool2d(rgb, scale)
            surf_xyz = surf_xyz[:, :, scale // 2::scale, scale // 2::scale]
            # free_xyz = free_xyz[:, scale//2::scale, scale//2::scale]
            _index = index.view(-1, 1, 1).expand(-1, ceil(H / scale), ceil(W / scale))

            if self.resnet:
                with torch.no_grad():
                    rgb = self.resnet(rgb)
            surf_features = self.rgb_conv_net(rgb)

        assert surf_xyz.size() == surf_rgb.size()

        if target_emb is not None:
            target_emb = self.pre_dot_layer(target_emb)
            target_emb = reduce(target_emb, "b f h w -> b f () ()", "mean")
            target_emb = torch.index_select(target_emb, 0, index)
            rgb_embedding = self.pre_dot_layer(F.pad(emb, pad=(2, 2, 2, 2)))
            dot_channel = reduce(target_emb * rgb_embedding, "n (f c) h w -> n c h w", "sum", f=self.h)
            dot_channel = self.post_dot_layer(dot_channel)
            surf_features = surf_features + dot_channel

        if valid_points is None:
            flat_features = rearrange(surf_features, "n f h w -> (n h w) f")
            flat_xyz = rearrange(surf_xyz, "n f h w -> (n h w) f")
            flat_rgb = rearrange(surf_rgb, "n f h w -> (n h w) f")
            # flat_free = rearrange(free_xyz, "n h w k f -> (n h w) k f")
            flat_index = rearrange(_index, "n h w -> (n h w)")
        else:
            if not self.use_embeddings:
                valid_points = valid_points[:, ::scale, ::scale]
            _surf_features = rearrange(surf_features, "n f h w -> n h w f")
            _surf_xyz = rearrange(surf_xyz, "n f h w -> n h w f")
            _surf_rgb = rearrange(surf_rgb, "n f h w -> n h w f")
            flat_features = _surf_features[valid_points]  # (M, F)
            flat_xyz = _surf_xyz[valid_points]  # (M, 3)
            flat_rgb = _surf_rgb[valid_points]  # (M, 3)
            # flat_free = free_xyz[_valid_points]     # (M, k, 3)
            flat_index = _index[valid_points]  # (M,)

        selector, samples = self.coord_to_grid(flat_xyz, flat_index, self.pcn_sample_ratio)
        flat_features, flat_rgb, selector = flat_features[samples], flat_rgb[samples], selector[samples]

        output_sizes = (batch_size, self.v_res, *self.map_res)

        feature_map, feature_counts = bin_reduce(flat_features, selector, output_sizes)
        rgb_map, _ = bin_reduce(flat_rgb, selector, output_sizes)

        # _flat_free = rearrange(flat_free, "m k f -> (m k) f")
        # _flat_index = flat_index.unsqueeze(1).repeat(1, K).view(-1)

        # selector, samples = self.coord_to_grid(_flat_free, _flat_index)
        # selector = selector[samples]
        # index = selector_to_index(selector, output_sizes)
        # free_map = get_counts(index, output_sizes)

        return {'feature_map': rearrange(feature_map, "b v x y f -> b f v x y"),
                'feature_counts': feature_counts.unsqueeze(1),
                'rgb_map': rearrange(rgb_map, "b v x y f -> b f v x y"),
                # 'free_map': free_map.unsqueeze(1)
                }

    def coord_to_grid(self, xyz, index, pcn_sample_ratio=1.0):
        """
        :param xyz: (M, 3)
        :param index: (M,)
        :return:
        """
        # convert coordinates to map grid indices
        map_h, map_w = self.map_res
        h1, w1, h2, w2 = self.map_bbox
        v1, v2 = self.v_bbox
        hs = ((xyz[:, self.xyz_to_h] - h1) * map_h / (h2 - h1)).long()  # (N',)
        ws = ((xyz[:, self.xyz_to_w] - w1) * map_w / (w2 - w1)).long()  # (N',)
        vs = ((xyz[:, self.xyz_to_v] - v1) * self.v_res / (v2 - v1)).long()  # (N',)
        selector = torch.stack([index, vs, hs, ws], dim=-1)
        samples = (hs >= 0) & (hs < map_h) & (ws >= 0) & (ws < map_w) & (vs >= 0) & (vs < self.v_res)

        if pcn_sample_ratio < 1.0:
            samples = samples & (torch.rand(len(selector), device=self.device) < self.pcn_sample_ratio)

        return selector, samples

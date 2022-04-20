# from pytorch_memlab import profile, set_target_gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from core.models.projection.point_cloud_to_2d import PointCloudTo2D


class PointCloudVINBase:
    def __init__(self, **config):
        super().__init__(**config)
        self.point_conv_net = self.get_point_conv_model(**config)

        self.reduce_thresh = config['reduce_thresh']
        self.h = h = config['pcn_h']  # size of hidden layers
        self.f = f = config['pcn_f']  # number of output features
        self.v_res = config['v_res']
        self.device = config['device']
        dropout = config['dropout']
        use_batch_norm = config['use_batch_norm']

        self.map_rf = 9  # Receptive field
        self.map_pad = self.map_rf // 2

        self.map_conv_net = nn.Sequential(
            # *((nn.GroupNorm(h+2, self.v_res * (h+2)),) if use_batch_norm else ()),
            nn.Conv3d(in_channels=h, out_channels=h, dilation=(1, 2, 2),
                      kernel_size=(self.v_res, 3, 3), stride=(1, 1, 1), padding=(0, 2, 2), bias=True),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv3d(in_channels=h, out_channels=f, dilation=(1, 2, 2),
                      kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 2, 2), bias=True)
        )

    def get_point_conv_model(self, **config):
        return PointCloudTo2D(**config)

    def __repr__(self):
        return f"{super().__repr__()}_{repr(self.point_conv_net)}_{self.f}"

    def forward(self, *, prev_map_raw=None, prev_counts_raw=None, prev_free_map_raw=None, prev_rgb_map_raw=None,
                new_episodes=None, inference=None, **kwargs):
        assert inference or (prev_map_raw is None and prev_counts_raw is None and new_episodes is None)
        outputs = self.point_conv_net(inference=inference, **kwargs)
        feature_map_raw, feature_counts_raw, rgb_map_raw, free_map_raw = \
            [outputs.get(k) for k in ["feature_map", "feature_counts", "rgb_map", "free_map"]]
        feature_map_raw = F.relu(feature_map_raw)

        if new_episodes is not None and not all(new_episodes):
            assert inference and prev_map_raw is not None and prev_counts_raw is not None and prev_rgb_map_raw is not None
            prev_map_raw[new_episodes] = 0
            prev_counts_raw[new_episodes] = 0
            feature_map_raw = feature_map_raw + prev_map_raw
            feature_counts_raw = feature_counts_raw + prev_counts_raw
            # free_map_raw = free_map_raw + prev_free_map_raw
            rgb_map_raw = rgb_map_raw + prev_rgb_map_raw

        feature_map = torch.where(feature_counts_raw > 0,
                                  feature_map_raw / feature_counts_raw,
                                  torch.zeros_like(feature_map_raw))
        rgb_map = torch.where(feature_counts_raw > 0,
                              rgb_map_raw / feature_counts_raw,
                              torch.zeros_like(rgb_map_raw))
        feature_counts = feature_counts_raw > self.reduce_thresh
        # free_map = free_map_raw > self.reduce_thresh

        feature_map = torch.cat([
            self.map_conv_net(feature_map).squeeze(2),
            rearrange(feature_counts.float(), "b f v x y -> b (f v) x y"),
            # free_map.float()
        ], dim=1)  # (b f v x y)
        # feature_map = feature_map, "b f v x y -> b (f v) x y")

        outputs = super().forward(feature_map=feature_map, inference=inference, new_episodes=new_episodes, **kwargs)
        return {
            **outputs,
            'feature_counts_raw': feature_counts_raw,
            'feature_map_raw': feature_map_raw,
            'feature_map': feature_map,
            'rgb_map_raw': rgb_map_raw,
            # 'free_map_raw': free_map_raw,
            'rgb_map': rgb_map,
            'feature_counts': feature_counts,
            # 'free_map': free_map,
        }


def add_pcn_train_args(parser):
    parser.add_argument('--pcn_i', type=int, default=3, help="point conv net input")
    parser.add_argument('--pcn_h', type=int, default=20)
    parser.add_argument('--pcn_f', type=int, default=20)
    parser.add_argument('--dot_channels', type=int, default=8)
    parser.add_argument('--reduce_thresh', '-rt', type=int, default=1)
    parser.add_argument('--pcn_sample_ratio', '-sr', type=float, default=1.0, help="point cloud sample ratio")
    parser.add_argument('--use_batch_norm', '-bn', action="store_true", help="use batch norm")
    parser.add_argument('--use_resnet', '-resnet', action='store_true', default=False)
    return parser

import torch
from torch import nn

from core.models.detector.vggnet.vgg_feature_extractor import PretrainedVGGShort


class VGGDetector(nn.Module):
    def __init__(self, h=64, device=None, **kwargs):
        super(VGGDetector, self).__init__()
        self.vggnet = PretrainedVGGShort(device=device, freeze=True)
        self.h = h

        self.rgb_conv_net = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=h, dilation=2,
                      kernel_size=3, stride=1, padding=2, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(in_channels=h, out_channels=h, dilation=2,
                      kernel_size=3, stride=1, padding=2, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(in_channels=h, out_channels=h, dilation=2,
                      kernel_size=3, stride=1, padding=2, bias=True),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(in_channels=h, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, images, **kwargs):
        with torch.no_grad():
            x = self.vggnet(images)
        t = self.rgb_conv_net(x)
        t = t.squeeze(1)
        t = torch.sigmoid(t)
        return {'t': t}

    def __repr__(self):
        return f"{self.__class__.__name__}_h_{self.h}"

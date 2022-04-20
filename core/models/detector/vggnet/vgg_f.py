import torch
import torch.nn as nn
import os


class Imagenet_matconvnet_vgg_f_dag(nn.Module):

    def __init__(self, device=None):
        super().__init__()
        self.meta = {'mean': [122.80329895019531, 114.88525390625, 101.57212829589844],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}

        self.mean = torch.tensor(self.meta['mean']).view((3, 1, 1)).to(device)
        self.std = torch.tensor(self.meta['std']).view((3, 1, 1)).to(device)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), ceil_mode=False)
        self.fc6 = nn.Conv2d(256, 4096, kernel_size=(6, 6), stride=(1, 1))
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, x):
        x0 = (x * 255 - self.mean) / self.std
        x1 = self.conv1(x0)
        x2 = self.relu1(x1)
        x3 = self.pool1(x2)
        x4 = self.conv2(x3)
        x5 = self.relu2(x4)
        x6 = self.pool2(x5)
        x7 = self.conv3(x6)
        x8 = self.relu3(x7)
        x9 = self.conv4(x8)
        x10 = self.relu4(x9)
        x11 = self.conv5(x10)
        x12 = self.relu5(x11)
        x13 = self.pool5(x12)
        x14 = self.fc6(x13)
        x15_preflatten = self.relu6(x14)
        x15 = x15_preflatten.view(x15_preflatten.size(0), -1)
        x16 = self.fc7(x15)
        x17 = self.relu7(x16)
        x18 = self.fc8(x17)
        return x18


def load_vgg_f(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Imagenet_matconvnet_vgg_f_dag()
    if weights_path is None: weights_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                         "imagenet_matconvnet_vgg_f_dag.pth")
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    return model

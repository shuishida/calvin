import torch
import torch.nn as nn
import torchvision.models as models


class PretrainedResNetShort(nn.Module):
    def __init__(self, device=None, freeze=True, cutoff_layers=4):
        super(PretrainedResNetShort, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*(list(resnet18.children())[:-cutoff_layers]))
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).to(device)
        self.to(device)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

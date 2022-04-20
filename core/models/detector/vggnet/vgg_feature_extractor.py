import torch.nn as nn

from core.models.detector.vggnet.vgg_f import load_vgg_f


class PretrainedVGGShort(nn.Module):
    def __init__(self, device=None, freeze=True):
        super(PretrainedVGGShort, self).__init__()
        vgg_f_model = load_vgg_f()
        self.model = nn.Sequential(*(list(vgg_f_model.children())[:-5]))
        print(self.model)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

from torch import nn
import torch.nn.functional as F

class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class ResNet50(nn.Module):
    def __init__(self, resnet, num_outputs):
        super(ResNet50, self).__init__()
        self.resnet = resnet
        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
            nn.Dropout(0.5),
            layer4
        )
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(2048, num_outputs)
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.resnet(x)
        return out

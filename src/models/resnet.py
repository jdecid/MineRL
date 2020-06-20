import torch
import torchvision
from torch import nn


class MResNet(nn.Module):
    def __init__(self, num_categorical: int, num_continuous: int):
        super().__init__()

        self.is_recurrent = False

        self.num_categorical = num_categorical
        self.num_continuous = num_continuous

        self.resnet = torchvision.models.resnet50(pretrained=True)

        num_filters = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_filters, out_features=num_categorical + 2 * num_continuous)
        # self.resnet.fc = nn.Linear(in_features=num_filters, out_features=8)

    def forward(self, x):
        out = self.resnet(x)
        out[:, 2 * self.num_continuous:] = torch.sigmoid(out[:, 2 * self.num_continuous:])
        # out[:, [2, 4, 6, 7]] = torch.sigmoid(out[:, [2, 4, 6, 7]])
        # out[:, [0, 1, 3, 5]] = torch.tanh(out[:, [0, 1, 3, 5]])
        return out

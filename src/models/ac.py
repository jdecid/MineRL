import torch
from torch import nn
from torch.nn import functional as F


class ValueCNNModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.cnn = model.cnn
        self.fc = nn.Linear(in_features=1600, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.cnn(x)
        x = x.view(batch_size, 1600)

        x = F.relu(x)
        return self.fc(x)
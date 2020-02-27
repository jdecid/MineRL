import torch.nn as nn

from src.models.cnn import PoVModel


class ImitationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = PoVModel()
        self.lstm = nn.LSTM(input_size=10 + 2, hidden_size=200)

    def forward(self, x):
        bs = x.size(0)
        x = self.cnn(x)
        x = x.view(bs, -1)
        x = self.fc(x)
        return x

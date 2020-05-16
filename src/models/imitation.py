import torch
import torch.nn as nn
from torch.nn import functional as F

from src.constants import HEIGHT, WIDTH, CHANNELS
from src.models.cnn import CNN


class ImitationLSTMModel(nn.Module):
    HIDDEN_LSTM_UNITS = 512

    def __init__(self, out_features: int, num_continuous: int):
        super().__init__()

        self.is_recurrent = True

        self.out_features = out_features
        self.num_continuous = num_continuous

        self.cnn = CNN()
        self.lstm = nn.LSTM(input_size=1600, hidden_size=self.HIDDEN_LSTM_UNITS, batch_first=True)
        self.fc = nn.Linear(in_features=self.HIDDEN_LSTM_UNITS, out_features=out_features)

    def forward(self, x, hc=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Flatten among batch and sequence dimensions
        x = x.view(-1, CHANNELS, WIDTH, HEIGHT)
        x = self.cnn(x)

        # Reshape back batch and sequence dimensions
        x = x.view(batch_size, seq_len, 1600)
        _, (h_n, c_n) = self.lstm(x, hc)

        x = F.relu(h_n)
        out = self.fc(x)

        # Set binary outputs in range [0, 1] to apply 0.5 threshold
        out[:, :, self.num_continuous:] = torch.sigmoid(out[:, :, self.num_continuous:])

        return out, (h_n, c_n)


class ImitationCNNModel(nn.Module):
    def __init__(self, out_features: int, num_continuous: int):
        super().__init__()

        self.is_recurrent = False

        self.out_features = out_features
        self.num_continuous = num_continuous

        self.cnn = CNN()
        self.fc = nn.Linear(in_features=1600, out_features=out_features)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.cnn(x)
        x = x.view(batch_size, 1600)

        x = F.relu(x)
        out = self.fc(x)

        # Set binary outputs in range [0, 1]
        out[:, self.num_continuous:] = torch.sigmoid(out[:, self.num_continuous:])

        # Set float outputs in range [-1, 1]
        out[:, :self.num_continuous] = torch.tanh(out[:, :self.num_continuous])

        return out

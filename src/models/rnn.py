import torch
import torch.nn as nn

from torch.nn import functional as F


class ImitationRNNModel(nn.Module):
    HIDDEN_LSTM_UNITS = 100
    OUTPUT_CNN_UNITS = 128

    def __init__(self, out_features: int, num_continuous: int):
        super().__init__()

        self.out_features = out_features
        self.num_continuous = num_continuous

        self.cnn = ImitationCNNModel(self.OUTPUT_CNN_UNITS)
        self.lstm = nn.LSTM(input_size=self.OUTPUT_CNN_UNITS, hidden_size=self.HIDDEN_LSTM_UNITS)
        self.fc = nn.Linear(in_features=self.HIDDEN_LSTM_UNITS, out_features=out_features)

    def forward(self, pov):
        seq_len = pov.size(0)

        x = self.cnn(pov)
        out, _ = self.lstm(x.view(seq_len, 1, self.OUTPUT_CNN_UNITS))

        out = F.relu(out)
        out = self.fc(out)

        # Set binary outputs in range [0, 1]
        out[:, :, self.num_continuous:] = torch.sigmoid(out[:, :, self.num_continuous:])

        return out


class ImitationCNNModel(nn.Module):
    def __init__(self, output_units: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(21632, 500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=output_units)
        )

    def forward(self, x):
        bs = x.size(0)
        x = self.cnn(x)
        x = x.view(bs, -1)
        x = self.fc(x)
        return x

import torch.nn as nn


class PoVModel(nn.Module):
    def __init__(self):
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
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=10)
        )

    def forward(self, x):
        bs = x.size(0)
        x = self.cnn(x)
        x = x.view(bs, -1)
        x = self.fc(x)
        return x

import gym
import minerl
from torch import nn


class ModelCNN(nn.Module):

    def __init__(self):
        self.conv = nn.Sequential(
            nn.Conv2d(3, 50, 3),
            nn.ReLU(),
            nn.Conv2d(3, 50, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(3, 50, 3),
            nn.ReLU(),
            nn.Conv2d(3, 50, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 50, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, X):
        BS = X.size(0)
        X = self.conv(X)
        X = X.view(BS)
        X = self.fc(X)


data = minerl.data.make("MineRLNavigate-v0", data_dir='/home/kayand/Documents/MineRL_data')

# Iterate through a single epoch using sequences of at most 32 steps
for current_state, action, reward, next_state, done \
    in data.sarsd_iter(num_epochs=1, max_sequence_len=32):
    print(current_state)
    print(action)



# Run a random agent through the environment
env = gym.make("MineRLNavigate-v0")  # A MineRLNavigate-v0 env

import gym
import numpy as np
import torch
from torch import nn
from typing import List

from collections import OrderedDict

# TODO: Generalize for N GPUs with DataParallel
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: Place¿?
BINARY_CONSTANTS = ['attack',
                    'back',
                    'forward',
                    'jump',
                    'left',
                    # 'place',
                    'right',
                    'sneak',
                    'sprint']


class ToTensor(object):
    def __call__(self, sample: np.ndarray, *args, **kwargs):
        tensor = torch.from_numpy(sample).to(torch.float)
        # tensor = 2 * (tensor / 255.0) - 1
        tensor = tensor / 255.0
        return tensor.permute(2, 0, 1)


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0)


def action_dict_to_tensor(x: OrderedDict, keys: List[str], contains: bool) -> torch.Tensor:
    # n = # keys matching
    # Output size (n)
    tensor = []

    for key in x.keys():
        if (contains and key in keys) or (not contains and key not in keys):
            value = x[key][0]

            if key != 'camera':
                tensor.append(value)
            else:
                tensor.append(value[0])
                tensor.append(value[1])

    return torch.tensor(tensor).to(torch.float).to(DEVICE)


def tensor_to_action_dict(env: gym.Env, x: torch.Tensor) -> OrderedDict:
    actions = env.action_space.noop()

    actions['camera'] = (float(x[0]), float(x[1]))
    for idx, action in enumerate(BINARY_CONSTANTS, start=2):
        actions[action] = int(x[idx] >= 0.5)

    return actions


class ImitationLoss(nn.Module):
    def __init__(self, num_continuous: int, writer):
        super().__init__()

        self.num_continuous = num_continuous
        self.writer = writer

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, y: List[OrderedDict], idx: int) -> torch.Tensor:
        categorical_targets = torch.stack(list(map(lambda x: action_dict_to_tensor(x, ['camera'], contains=False), y)))
        categorical_loss = self.bce_criterion(pred[:, self.num_continuous:], categorical_targets)

        mse_targets = torch.stack(list(map(lambda x: action_dict_to_tensor(x, ['camera'], contains=True), y)))
        mse_loss = self.mse_criterion(pred[:, :self.num_continuous], mse_targets)

        loss = categorical_loss + MSE_MULTIPLIER * mse_loss

        self.writer.add_scalars('Loss', {
            'Categorical Loss': categorical_loss.item(),
            'MSE Loss': MSE_MULTIPLIER * mse_loss.item()
        }, global_step=idx)

        return loss


MSE_MULTIPLIER = 0.01

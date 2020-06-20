from collections import OrderedDict
from typing import List, Tuple

import gym
import numpy as np
import torch
from torch import nn
# TODO: Generalize for N GPUs with DataParallel
from torch.distributions import Normal, Bernoulli

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BINARY_CONSTANTS = ['attack',
                    'back',
                    'forward',
                    'jump',
                    'left',
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


def action_dict_to_tensor_v2(x: OrderedDict, keys: List[str], contains: bool) -> torch.Tensor:
    tensor = [
        x['camera'][0][0] / 180,
        x['camera'][0][1] / 180
    ] if (contains and 'camera' in keys) or (not contains and 'camera' not in keys) else []

    idx = 0
    while idx < len(BINARY_CONSTANTS):
        action_name = BINARY_CONSTANTS[idx]
        if (contains and action_name in keys) or (not contains and action_name not in keys):
            if action_name in ['back', 'forward', 'left', 'right']:
                action_name_2 = BINARY_CONSTANTS[idx + 1]

                if x[action_name][0] == 1:
                    tensor.append(-1)
                elif x[action_name_2][0] == 1:
                    tensor.append(1)
                else:
                    tensor.append(0)

                idx += 1
            else:
                tensor.append(x[action_name][0])

        idx += 1

    return torch.tensor(tensor).to(torch.float).to(DEVICE)


def tensor_to_action_dict(env: gym.Env, x: torch.Tensor) -> OrderedDict:
    """
    Transforms network output tensor into an action dictionary for the Gym environment.
    The camera parameters are set directly with the provided values.
    The rest of actions are binarized using a threshold value of 0.5.
    :param env: Environment
    :param x: Network output tensor of size (12,)
        - The first four elements correspond to the camera parameters:
            - x0 and x2 are the mean and std of the X-axis, respectively.
            - x1 and x3 are the mean and std of the Y-axis, respectively.
        - Parameters from x4 to x11, in [0, 1] correspond to the actions (ordered):
            'attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint'
    :return: Action dictionary with the following keys:
        {'camera', 'attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint'}
    """
    actions = env.action_space.noop()

    actions['camera'] = (float(x[0]), float(x[1]))
    for idx, action in enumerate(BINARY_CONSTANTS, start=4):
        actions[action] = int(x[idx] >= 0.5)

    return actions


def tensor_to_action_dict_v2(env: gym.Env, x: torch.Tensor) -> OrderedDict:
    """
    Transforms network output tensor into an action dictionary for the Gym environment.
    The camera parameters are set directly with the provided values.
    The other actions in [0, 1] are binarized using a threshold value of 0.5.
    The rest of actions (in [-1, 1]) are described in the specific parameter explanation.
    :param env: Environment
    :param x: Network output tensor of size (8,)
        - The first two elements correspond to the camera parameters, in [-1, 1]. For a proper conversion from discrete
          back to continuous variables, instead of mapping {-1: -180, 0: 0, 1: 180}, the angle is lineal to the input.
            - x0 is the discretized camera X-axis in respectively.
            - x1 is the discretized camera Y-axis in respectively.
        - Parameters from x2 to x8 correspond to the actions (ordered).
            - x2 ∈ [0, 1] corresponds to the 'attack' parameter.
            - x3 ∈ [-1, 1] corresponds to the 'back' and 'forward' parameters. Setting a neutral zone in [-0.33, 0.33],
              where neither of the frontal actions are performed, negative values activate `back` while positive do the
              same for the `forward` action.
            - x4 ∈ [0, 1] corresponds to the 'jump' parameter.
            - x5 ∈ [-1, 1] corresponds to the 'left' and 'right' parameters. Setting a neutral zone in [-0.33, 0.33],
              where neither of the lateral actions are performed, negative values activate `left` while positive do
              the same for the `right` action.
            - x6 ∈ [0, 1] corresponds to the 'sneak' parameter.
            - x7 ∈ [0, 1] corresponds to the 'sprint' parameter.
    :return: Action dictionary with the following keys:
        {'camera', 'attack', 'back', 'forward', 'jump', 'left', 'right', 'sneak', 'sprint'}
    """
    actions = env.action_space.noop()

    actions['camera'] = (180 * float(x[0]), 180 * float(x[1]))

    idx = 0
    tensor_idx = 2
    while idx < len(BINARY_CONSTANTS):
        action_name = BINARY_CONSTANTS[idx]
        if idx == 1 or idx == 4:
            # Negative action
            actions[action_name] = int(x[tensor_idx] < -0.33)
            # Positive action
            action_name = BINARY_CONSTANTS[idx + 1]
            actions[action_name] = int(x[tensor_idx] > 0.33)
            idx += 1
        else:
            actions[action_name] = int(x[tensor_idx] >= 0.5)

        idx += 1
        tensor_idx += 1

    return actions


def tensor_to_probabilistic_action_dict(env: gym.Env, x: torch.Tensor) -> Tuple[OrderedDict, torch.Tensor]:
    actions = env.action_space.noop()
    log_probs = [0 for _ in range(10)]

    m_1 = Normal(x[0], x[2])
    camera_action_1 = m_1.rsample()
    log_prob_1 = m_1.log_prob(camera_action_1)
    log_probs[0] = log_prob_1 if not torch.isnan(log_prob_1).any() else torch.tensor(0.0, device=DEVICE)

    m_2 = Normal(x[1], x[3])
    camera_action_2 = m_2.rsample()
    log_prob_2 = m_2.log_prob(camera_action_2)
    log_probs[1] = log_prob_2 if not torch.isnan(log_prob_2).any() else torch.tensor(0.0, device=DEVICE)

    actions['camera'] = (float(camera_action_1.item()), float(camera_action_2.item()))

    for idx, action in enumerate(BINARY_CONSTANTS, start=4):
        m = Bernoulli(x[idx])
        sampled_action = m.sample()
        log_probs[idx - 2] = m.log_prob(sampled_action)
        actions[action] = int(sampled_action)

    # print(log_probs)

    return actions, torch.stack(log_probs).sum()


class ImitationLoss(nn.Module):
    def __init__(self, num_continuous: int, writer):
        super().__init__()

        self.num_continuous = num_continuous
        self.writer = writer

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, y: List[OrderedDict], idx: int) -> torch.Tensor:
        categorical_targets = torch.stack(list(map(lambda x: action_dict_to_tensor(x, ['camera'], contains=False), y)))
        categorical_loss = self.bce_criterion(pred[:, 2 * self.num_continuous:], categorical_targets)

        mse_targets = torch.stack(list(map(lambda x: action_dict_to_tensor(x, ['camera'], contains=True), y)))
        std = mse_targets.std(dim=0)
        std = torch.cat(mse_targets.size(0) * [std.unsqueeze(0)], 0)
        mse_loss = self.mse_criterion(pred[:, :self.num_continuous], mse_targets)
        mse_loss += self.mse_criterion(pred[:, self.num_continuous:2 * self.num_continuous], std)

        loss = categorical_loss + MSE_MULTIPLIER * mse_loss

        self.writer.add_scalars('Loss', {
            'Categorical Loss': categorical_loss.item(),
            'MSE Loss': MSE_MULTIPLIER * mse_loss.item()
        }, global_step=idx)

        return loss


class ImitationLossV2(nn.Module):
    def __init__(self, writer):
        super().__init__()

        self.writer = writer

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, y: List[OrderedDict], idx: int) -> torch.Tensor:
        categorical_targets = torch.stack(list(map(lambda x: action_dict_to_tensor_v2(x, ['camera', 'back', 'forward',
                                                                                          'left', 'right'],
                                                                                      contains=False), y)))

        categorical_loss = self.bce_criterion(pred[:, [2, 4, 6, 7]], categorical_targets)

        mse_targets = torch.stack(
            list(map(
                lambda x: action_dict_to_tensor_v2(x, ['camera', 'back', 'forward', 'left', 'right'], contains=True),
                y)))
        mse_loss = self.mse_criterion(pred[:, [0, 1, 3, 5]], mse_targets)

        loss = categorical_loss + mse_loss

        self.writer.add_scalars('Loss', {
            'Categorical Loss': categorical_loss.item(),
            'MSE Loss': mse_loss.item()
        }, global_step=idx)

        return loss


MSE_MULTIPLIER = 0.01

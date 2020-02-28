import numpy as np
import torch
from torch import nn
from typing import List

from collections import OrderedDict

# TODO: Generalize for N GPUs with DataParallel
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: Place¿?
BINARY_CONSTANTS = ["attack",
                    "back",
                    "forward",
                    "jump",
                    "left",
                    # "place",
                    "right",
                    "sneak",
                    "sprint"]


class ToTensor(object):
    def __call__(self, sample: np.ndarray, *args, **kwargs):
        dims = len(sample.shape)
        assert dims == 3 or dims == 4, 'Matrix must have 3 or 4 dimensions'

        tensor = torch.from_numpy(sample).to(torch.float)
        # TODO: Change with Normalize (0, 1) outside
        tensor = 2 * (tensor / 255.0) - 1

        if dims == 4:
            return tensor.permute(0, 3, 1, 2)
        else:
            return tensor.permute(2, 0, 1)


def action_dict_to_tensor(x: OrderedDict, keys: List[str], contain: bool) -> torch.Tensor:
    tensor = []

    for key in x.keys():
        if (contain and key in keys) or (not contain and key not in keys):
            value = x[key]

            if key != 'camera':
                tensor.append(torch.tensor(value))
            else:
                tensor.append(torch.tensor(value[0]))
                tensor.append(torch.tensor(value[1]))

    return torch.stack(tensor).permute(1, 0).to(torch.float).to(DEVICE)


def tensor_to_action_dict(x: torch.Tensor) -> OrderedDict:
    actions = OrderedDict()

    actions['camera'] = x[0, 0, :2]
    actions['place'] = 'none'

    print(x.size())

    for idx, action in enumerate(BINARY_CONSTANTS, start=2):
        actions[action] = int(x[0, 0, idx] >= 0.5)

    return actions


class ImitationLoss(nn.Module):
    def __init__(self, num_continuous: int):
        super().__init__()

        self.num_continuous = num_continuous

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, y: OrderedDict) -> torch.Tensor:
        categorical_targets = list(map(lambda x: action_dict_to_tensor(x, ['camera'], contain=False), y))
        categorical_targets = torch.stack(categorical_targets)[:, -1, :]
        categorical_loss = self.bce_criterion(pred[:, self.num_continuous:], categorical_targets)

        mse_targets = list(map(lambda x: action_dict_to_tensor(x, ['camera'], contain=True), y))
        mse_targets = torch.stack(mse_targets)[:, -1, :]
        mse_loss = self.mse_criterion(pred[:, :self.num_continuous], mse_targets)

        return categorical_loss + mse_loss

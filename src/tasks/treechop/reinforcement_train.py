import os
import uuid

import gym
import numpy as np
import torch

from src.models.imitation import PolicyCNNModel, PolicyLSTMModel
from src.tasks.treechop.imitation_train import transformation
from src.utils import DEVICE, tensor_to_probabilistic_action_dict


def load_model(model_type: str):
    if model_type == 'CNN':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], 'ImitationCNNModel_X.pt')
    else:
        model = PolicyLSTMModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], 'ImitationLSTMModel_6.pt')

    model.load_state_dict(torch.load(path))
    model.to(DEVICE)

    return model


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma ** i * r for i, r in enumerate(rewards)])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def main(model: torch.nn.Module, n_envs: int, run_timestamp: str):
    uid = str(uuid.uuid4())
    file_path = os.path.join('results', 'treechop', 'rewards', 'LSTM' if model.is_recurrent else 'CNN', f'{uid}.csv')
    print(f'Saving {uid}')

    repeats_rewards = []

    env = gym.make('MineRLTreechop-v0')

    buffer = REINFORCEBuffer()

    for i in range(n_envs):
        print(f'Execution {i}')
        rewards = run_env(model, buffer, env)
        buffer.update_rewards(rewards)

        repeats_rewards.append(rewards)

    with open(file_path, mode='w') as f:
        f.writelines([','.join(rewards) + '\n' for rewards in repeats_rewards])


class REINFORCEBuffer:
    def __init__(self):
        self.batch_counter = 0
        self.batch_rewards = []
        self.batch_states = []
        self.batch_actions = []
        self.total_rewards = []

    def update_state_actions(self, states, actions):
        self.batch_states.append(states)
        self.batch_actions.append(actions)

    def update_rewards(self, rewards):
        self.batch_rewards += discount_rewards(rewards)
        self.total_rewards.append(sum(rewards))
        self.batch_counter += 1

    def get_loss(self):
        tensor_states = torch.tensor()
        tensor_actions = self.batch_actions
        tensor_rewards = self.batch_rewards

        logprob = torch.log(policy_estimator.predict(torch.tensor(self.batch_states)))
        selected_logprobs = torch.tensor(self.batch_rewards) * torch.gather(logprob, 1,
                                                                            torch.tensor(self.batch_actions)).squeeze()
        loss = -selected_logprobs.mean()
        return loss


def run_env(model, buffer, env):
    obs = env.reset()
    done = False
    hc = None

    idx = 0
    rewards = []
    while not done and idx < 1000:
        frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)

        if not model.is_recurrent:
            pred = model(frame)
        else:
            pred, hc = model(frame.unsqueeze_(0), hc)

        action = tensor_to_probabilistic_action_dict(env, pred.squeeze())

        new_obs, rew, done, _ = env.step(action)
        rewards.append(str(rew))

        buffer.update_state_actions(rew, obs, action)

        obs = new_obs
        idx += 1

    return rewards

import os
import uuid

import gym
import numpy as np
import torch

from src.tasks.treechop.imitation_train import transformation
from src.utils import DEVICE, tensor_to_probabilistic_action_dict, tensor_to_action_dict


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma ** i * r for i, r in enumerate(rewards)])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def main(model: torch.nn.Module, n_envs: int, run_timestamp: str):
    uid = str(uuid.uuid4())
    file_path = os.path.join('results', 'treechop', 'rewards', 'LSTM' if model.is_recurrent else 'CNN', f'{uid}.csv')
    print(f'Saving {uid}')

    repeats_rewards = []

    # env = gym.make('MineRLTreechop-v0')

    optimizer = torch.optim.Adam(model.parameters())

    for i in range(n_envs):
        print(f'Execution {i}')
        # results = run_env(model, env)
        test_function(model, optimizer)

        # optimize_model(optimizer, results)

        # repeats_rewards.append(results['rewards'])

    with open(file_path, mode='w') as f:
        f.writelines([','.join(list(map(str, rewards))) + '\n' for rewards in repeats_rewards])


def optimize_model(optimizer: torch.optim.Optimizer, results):
    long_term_reward = discount_rewards(results['rewards'])
    for t in range(len(long_term_reward)):
        optimizer.zero_grad()

        loss = results['log_probs'][t]  # * long_term_reward[t]
        loss.backward()

        optimizer.step()


def run_env(model, env):
    obs = env.reset()
    done = False
    hc = None

    idx = 0

    log_probs = []
    rewards = []

    while not done and idx < 100:
        frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)

        if not model.is_recurrent:
            pred = model(frame)
        else:
            pred, hc = model(frame.unsqueeze_(0), hc)

        action, log_prob = tensor_to_probabilistic_action_dict(env, pred.squeeze())
        # action = tensor_to_action_dict(env, pred.squeeze())

        new_obs, rew, done, _ = env.step(action)

        rewards.append(rew)
        log_probs.append(log_prob)

        obs = new_obs
        idx += 1

    return {
        'rewards': rewards,
        'log_probs': log_probs
    }


def test_function(model, optimizer):
    pred = model(torch.rand((1, 3, 64, 64), device=DEVICE))
    loss = pred[0].sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

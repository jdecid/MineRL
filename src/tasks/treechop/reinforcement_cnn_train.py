import os

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.tasks.treechop.imitation_train import transformation
from src.utils import DEVICE, tensor_to_probabilistic_action_dict


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma ** i * r for i, r in enumerate(rewards)])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def main(model: torch.nn.Module, episodes: int, iterations: int, run_timestamp: str):
    log_dir = os.path.join(os.environ['LOGS_DIR'], run_timestamp)
    checkpoint_dir = os.path.join(os.environ['CHECKPOINT_DIR'], run_timestamp)
    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)

    writer = SummaryWriter(log_dir=log_dir)

    repeats_rewards = []

    env = gym.make('MineRLTreechop-v0')

    optimizer = torch.optim.Adam(model.parameters())

    for i in range(episodes):
        print(f'Episode {i}')
        results = run_env(model, env, iterations)

        optimize_model(optimizer, results, writer, i)

        repeats_rewards.append(results['rewards'])


def optimize_model(optimizer: torch.optim.Optimizer, results, writer, idx):
    long_term_reward = discount_rewards(results['rewards'])

    optimizer.zero_grad()

    loss = torch.zeros(1, device=DEVICE)
    for t in range(len(long_term_reward)):
        loss += results['log_probs'][t] * long_term_reward[t]

    loss.backward()
    optimizer.step()

    cumsum_reward = torch.cumsum(torch.tensor(results['rewards']), dim=0)

    writer.add_scalars('Loss', {
        'Loss': loss.item(),
        'Mean Reward': cumsum_reward.mean().item(),
        'Final Reward': cumsum_reward[-1].item()
    }, global_step=idx)


def run_env(model, env, iterations: int):
    obs = env.reset()
    done = False
    hc = None

    idx = 0

    log_probs = []
    rewards = []

    while not done and idx < iterations:
        frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)

        if not model.is_recurrent:
            pred = model(frame)
        else:
            pred, hc = model(frame.unsqueeze_(0), hc)

        action, log_prob = tensor_to_probabilistic_action_dict(env, pred[0])
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


def test_function(model, optimizer, frame):
    pred = model(frame)
    loss = pred[0].sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("It works!")

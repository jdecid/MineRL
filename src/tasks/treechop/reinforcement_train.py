import os

import gym
import numpy as np
import torch
from torch.distributions import Bernoulli
from torch.utils.tensorboard import SummaryWriter

from src.tasks.treechop.imitation_train import transformation, SEQ_LEN
from src.utils import DEVICE, tensor_to_probabilistic_action_dict, tensor_to_action_dict


def discount_rewards(rewards, gamma=0.9):
    dr = []
    for t in range(len(rewards)):
        r = np.array([gamma ** i * r for i, r in enumerate(rewards[t:])])
        dr.append(r[::-1].cumsum()[-1])
    return dr


def main(model: torch.nn.Module, episodes: int, iterations: int, eps: float, reset_step: bool, run_timestamp: str):
    model.train()

    log_dir = os.path.join(os.environ['LOGS_DIR'], run_timestamp)
    checkpoint_dir = os.path.join(os.environ['CHECKPOINT_DIR'], run_timestamp)
    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)

    writer = SummaryWriter(log_dir=log_dir)

    env = gym.make('MineRLTreechop-v0')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for idx in range(episodes):
        print(f'Episode {idx}')
        results = run_env(model, env, iterations, eps, reset_step)

        optimize_model(optimizer, results, writer, idx)

        if idx % 10 == 0:
            print(f'Saving checkpoint {idx // 10}')
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f'{model.__class__.__name__}_{idx // 10}.pt'))


def optimize_model(optimizer: torch.optim.Optimizer, results, writer, idx):
    long_term_reward = discount_rewards(results['rewards'])
    print('[LTR] ', long_term_reward)

    optimizer.zero_grad()

    loss = torch.zeros(1, device=DEVICE)
    for t in range(len(long_term_reward)):
        loss += results['log_probs'][t] * long_term_reward[t]

    print(results['rewards'])
    print(loss)

    loss.backward()
    optimizer.step()

    cumsum_reward = torch.cumsum(torch.tensor(results['rewards']), dim=0)

    writer.add_scalars('Loss', {
        'Loss': loss.item(),
        'Mean Reward': cumsum_reward.mean().item(),
        'Final Reward': cumsum_reward[-1].item()
    }, global_step=idx)


def run_env(model, env, iterations: int, eps: float, reset_step=False):
    obs = env.reset()
    done = False
    hc = None

    idx = 0

    sequence_frames = []
    log_probs = []
    rewards = []

    bernoulli = Bernoulli(eps)

    while not done and idx < iterations:
        frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)
        sequence_frames.append(frame)
        if len(sequence_frames) > SEQ_LEN:
            sequence_frames = sequence_frames[-SEQ_LEN:]

        if not model.is_recurrent:
            pred = model(frame)
        else:
            if reset_step:
                pred, _ = model(torch.cat(sequence_frames, 0).unsqueeze_(0))
            else:
                pred, hc = model(frame.unsqueeze_(0), hc)

        if int(bernoulli.sample().item()):
            action = tensor_to_action_dict(env, pred.squeeze())
            log_prob = torch.log(torch.tensor(eps, device=DEVICE))
            print(f'Det: {log_prob}')
        else:
            action, log_prob = tensor_to_probabilistic_action_dict(env, pred.squeeze())
            log_prob += torch.log(torch.tensor(1 - eps, device=DEVICE))
            print(f'Prob: {log_prob}')

        print('-' * 20)

        new_obs, rew, done, _ = env.step(action)

        rewards.append(rew)
        log_probs.append(log_prob)

        obs = new_obs
        idx += 1

    return {
        'rewards': rewards,
        'log_probs': log_probs
    }

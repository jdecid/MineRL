import os

import gym
import minerl
import numpy as np
import torch
from torch.distributions import Bernoulli
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

from src.tasks.treechop.imitation_train import transformation, SEQ_LEN
from src.utils import DEVICE, tensor_to_probabilistic_action_dict, tensor_to_action_dict

REG = 0


def discount_rewards(rewards, gamma=0.9):
    dr = []
    for t in range(len(rewards)):
        r = np.array([gamma ** i * r for i, r in enumerate(rewards[t:])])
        dr.append(r[::-1].cumsum()[-1])
    return dr


def next_batch(data, epochs: int, batch_size: int):
    batch_frames = []
    batch_rewards = []

    for current_state, _, rew, _, _ in data.sarsd_iter(num_epochs=epochs, max_sequence_len=100):
        frames = current_state['pov']
        frames = torch.stack(list(map(transformation, frames)))

        long_term_rewards = discount_rewards(rew)

        batch_frames += frames[:50]
        batch_rewards += long_term_rewards[:50]

        if len(batch_frames) > batch_size:
            batch_frames = torch.stack(batch_frames)
            yield batch_frames.squeeze(), torch.tensor(batch_rewards).float()

            batch_frames = []
            batch_rewards = []

    if len(batch_frames) > 0:
        batch_frames = torch.stack(batch_frames)
        yield batch_frames.squeeze(), torch.tensor(batch_rewards).float()


def pretrain_value_model(model: torch.nn.Module, optimizer: torch.optim, epochs=1, single_batch=False):
    print('Pretraining Value model')

    minerl.data.download(os.environ['DATASET_DIR'], experiment='MineRLTreechop-v0')
    data = minerl.data.make('MineRLTreechop-v0', data_dir=os.environ['DATASET_DIR'])

    criterion = MSELoss()

    for idx, (frames, target_rewards) in enumerate(next_batch(data, epochs, 512), start=1):
        frames = frames.to(DEVICE)
        target_rewards = target_rewards.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        prediction = model(frames)

        loss = criterion(prediction.squeeze(), target_rewards)
        loss.backward()

        optimizer.step()

        if single_batch:
            return


def main(policy_model: torch.nn.Module, value_model: torch.nn.Module,
         episodes: int, iterations: int, eps: float, reset_step: bool, reg: float, run_timestamp: str):
    policy_model.train()
    value_model.train()

    global REG
    REG = reg

    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-4)
    value_optimizer = torch.optim.Adam(value_model.parameters())

    pretrain_value_model(value_model, value_optimizer, epochs=2)

    log_dir = os.path.join(os.environ['LOGS_DIR'], run_timestamp, 'AC_CNN')
    checkpoint_dir = os.path.join(os.environ['CHECKPOINT_DIR'], run_timestamp)
    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)

    writer = SummaryWriter(log_dir=log_dir)

    env = gym.make('MineRLTreechop-v0')

    for idx in range(episodes):
        print(f'Episode {idx}')
        results = run_env(policy_model, env, iterations, eps, reset_step)

        optimize_model(value_model, policy_optimizer, value_optimizer, results, writer, idx)

        if idx % 10 == 0:
            print(f'Saving checkpoint {idx // 10}')
            torch.save(policy_model.state_dict(),
                       os.path.join(checkpoint_dir, f'{policy_model.__class__.__name__}_{idx // 10}.pt'))
            torch.save(value_model.state_dict(),
                       os.path.join(checkpoint_dir, f'{value_model.__class__.__name__}_{idx // 10}.pt'))

        pretrain_value_model(value_model, value_optimizer, single_batch=True)


def optimize_model(value_model: torch.nn.Module, policy_optimizer: torch.optim.Optimizer,
                   value_optimizer: torch.optim.Optimizer, results, writer, idx):
    long_term_reward = discount_rewards(results['rewards'])
    print('[LTR] ', long_term_reward)

    delta = [long_term_reward[t] - value_model(results['frames'][t]).squeeze() for t in range(len(long_term_reward))]

    policy_optimizer.zero_grad()

    loss = torch.zeros(1, device=DEVICE)
    for t in range(len(long_term_reward)):
        loss += results['log_probs'][t] * delta[t]
        loss += results['log_probs'][t] * REG

    print(results['rewards'])
    print(loss)

    loss.backward(retain_graph=True)
    policy_optimizer.step()

    value_optimizer.zero_grad()

    value_loss = torch.sum(torch.stack(delta) ** 2)
    value_loss.backward()

    value_optimizer.step()

    cumsum_reward = torch.cumsum(torch.tensor(results['rewards']), dim=0)

    print(delta)

    writer.add_scalars('Loss', {
        'Loss': loss.item(),
        'Value': value_loss.item(),
        'Mean Reward': cumsum_reward.mean().item(),
        'Final Reward': cumsum_reward[-1].item()
    }, global_step=idx)


def run_env(model, env, iterations: int, eps: float, reset_step=False):
    obs = env.reset()
    done = False
    hc = None

    idx = 0

    all_frames = []
    sequence_frames = []
    log_probs = []
    rewards = []

    bernoulli = Bernoulli(eps)

    while not done and idx < iterations:
        frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)
        sequence_frames.append(frame)
        if len(sequence_frames) > SEQ_LEN:
            sequence_frames = sequence_frames[-SEQ_LEN:]
        all_frames.append(frame)

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
        'frames': all_frames,
        'rewards': rewards,
        'log_probs': log_probs
    }

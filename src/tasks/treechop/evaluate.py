import argparse
import os
import uuid

import gym
import torch
from dotenv import load_dotenv

from src.models.imitation import PolicyCNNModel, PolicyLSTMModel
from src.tasks.treechop.imitation_train import transformation, SEQ_LEN
from src.utils import tensor_to_action_dict, DEVICE


def load_model(model_type: str):
    if model_type == 'CNN':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-08_23-05-45-108802', 'PolicyCNNModel_11.pt')
    else:
        model = PolicyLSTMModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-08_17-14-00-858925', 'PolicyLSTMModel_6.pt')

    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()

    return model


def eval_model(model: torch.nn.Module, n_envs: int):
    uid = str(uuid.uuid4())
    file_path = os.path.join('results', 'treechop', 'rewards', 'LSTM' if model.is_recurrent else 'CNN', f'{uid}.csv')
    print(f'Saving {uid}')

    repeats_rewards = []

    env = gym.make('MineRLTreechop-v0')

    for i in range(n_envs):
        print(f'Execution {i}')
        rewards = run_env(model, env)
        repeats_rewards.append(rewards)

    with open(file_path, mode='w') as f:
        f.writelines([','.join(rewards) + '\n' for rewards in repeats_rewards])


def run_env(model, env):
    obs = env.reset()
    done = False
    hc = None

    frames = []

    with torch.no_grad():
        idx = 0
        rewards = []
        while not done and idx < 1000:
            frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)
            frames.append(frame)
            if len(frames) > SEQ_LEN:
                frames = frames[-SEQ_LEN:]

            if not model.is_recurrent:
                pred = model(frame)
            else:
                # pred, hc = model(frame.unsqueeze_(0), hc)
                pred, _ = model(torch.cat(frames, 0).unsqueeze_(0))

            action = tensor_to_action_dict(env, pred.squeeze())
            # print(action)

            obs, rew, done, _ = env.step(action)
            rewards.append(str(rew))

            idx += 1

    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['CNN', 'LSTM'])
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    eval_model(load_model(args.model), n_envs=args.repeats)

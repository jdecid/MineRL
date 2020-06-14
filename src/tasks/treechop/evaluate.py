import argparse
import os
import uuid

import gym
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from torch.distributions import Bernoulli

from src.models.ac import ValueCNNModel
from src.models.imitation import PolicyCNNModel, PolicyLSTMModel
from src.tasks.treechop.imitation_train import transformation, SEQ_LEN
from src.utils import tensor_to_action_dict, DEVICE, tensor_to_probabilistic_action_dict


def load_model(model_type: str):
    if model_type == 'CNN':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-08_23-05-45-108802', 'PolicyCNNModel_11.pt')
    elif model_type == 'Value':
        model = ValueCNNModel(PolicyCNNModel(num_categorical=8, num_continuous=2))
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-14_18-09-18-321950', 'ValueCNNModel_0.pt')
    elif model_type == 'REINFORCE_CNN':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-14_10-07-03-179864', 'PolicyCNNModel_3.pt')
    elif model_type == 'REINFORCE_LSTM':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        # TODO: TRAIN BOY
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-14_11-18-16-829177', 'PolicyLSTMModel_0.pt')
    elif model_type == 'AC_CNN':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        # TODO: TRAIN BOY
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-14_11-18-16-829177', 'PolicyLSTMModel_0.pt')
    else:
        model = PolicyLSTMModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-09_19-53-32-296107', 'PolicyLSTMModel_4.pt')

    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()

    return model


def generate_gif(images, output):
    save_path = 'results'

    img, *imgs = [Image.fromarray(i, 'RGB') for i in images]
    for idx, i in enumerate(imgs):
        i.save(os.path.join(save_path, f'{idx}.png'))

    img.save(fp=output, format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)


def eval_model(model: torch.nn.Module, n_envs: int, iterations: int, reset_step: bool):
    uid = str(uuid.uuid4())
    file_path = os.path.join('results', 'treechop', 'rewards', 'LSTM' if model.is_recurrent else 'CNN', f'{uid}.csv')
    print(f'Saving {uid}')

    repeats_rewards = []

    env = gym.make('MineRLTreechop-v0')

    for i in range(n_envs):
        print(f'Execution {i}')
        rewards = run_env(model, env, iterations, reset_step)
        repeats_rewards.append(rewards)

    with open(file_path, mode='w') as f:
        f.writelines([','.join(rewards) + '\n' for rewards in repeats_rewards])


def run_env(model, env, iterations: int, reset_step: bool, eps=1.0):
    obs = env.reset()
    done = False
    hc = None

    sequence_frames = []
    all_frames = []

    bernoulli = Bernoulli(eps)

    with torch.no_grad():
        idx = 0
        rewards = []
        while not done and idx < iterations:
            frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)
            sequence_frames.append(frame)
            all_frames.append((255 * frame.cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8))

            if len(sequence_frames) > SEQ_LEN:
                sequence_frames = sequence_frames[-SEQ_LEN:]

            if not model.is_recurrent:
                pred = model(frame)
            else:
                if reset_step:
                    pred, _ = model(torch.cat(sequence_frames, 0).unsqueeze_(0))
                else:
                    pred, hc = model(frame.unsqueeze_(0), hc)

            if eps == 1.0:
                action = tensor_to_action_dict(env, pred.squeeze())
            else:
                if int(bernoulli.sample().item()):
                    action = tensor_to_action_dict(env, pred.squeeze())
                else:
                    action, _ = tensor_to_probabilistic_action_dict(env, pred.squeeze())

            obs, rew, done, _ = env.step(action)
            rewards.append(str(rew))

            idx += 1

    # save = input('Save GIF? ')
    # if save == 'Y':
    #    generate_gif(all_frames, 'output.gif')

    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['CNN', 'LSTM'])
    parser.add_argument('--reset_step', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1000)
    args = parser.parse_args()

    load_dotenv()
    eval_model(load_model(args.model), n_envs=args.episodes, iterations=args.iterations, reset_step=args.reset_step)

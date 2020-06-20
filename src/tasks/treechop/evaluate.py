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
from src.models.resnet import MResNet
from src.tasks.treechop.imitation_train import transformation, SEQ_LEN
from src.utils import tensor_to_action_dict, DEVICE, tensor_to_probabilistic_action_dict

SAVED_GIF_COUNT = 0


def load_model(model_type: str):
    if model_type == 'CNN':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], 'CNN_BASE_IMITATION', 'PolicyCNNModel_11.pt')
    elif model_type == 'LSTM':
        model = PolicyLSTMModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], 'LSTM_LARGE_V2', 'PolicyLSTMModel_6.pt')
    elif model_type == 'Value':
        model = ValueCNNModel(PolicyCNNModel(num_categorical=8, num_continuous=2))
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-14_18-09-18-321950', 'ValueCNNModel_0.pt')
    elif model_type == 'ResNet':
        model = MResNet(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], 'RESNET_V1', 'MResNet_6.pt')
    elif model_type == 'REINFORCE_CNN':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-17_09-26-36-985449', 'PolicyCNNModel_9.pt')
    elif model_type == 'REINFORCE_LSTM':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        # TODO: TRAIN BOY
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-17_09-26-36-985449', 'PolicyLSTMModel_0.pt')
    elif model_type == 'AC_CNN':
        model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        # TODO: TRAIN BOY
        path = os.path.join(os.environ['CHECKPOINT_DIR'], 'AC_CNN', 'PolicyCNNModel_9.pt')
    else:
        model = PolicyLSTMModel(num_categorical=8, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], '2020-06-09_19-53-32-296107', 'PolicyLSTMModel_4.pt')

    # TODO: RESNET (?)
    # if model_type == 'CNN':
    #     # model = MResNet(num_categorical=8, num_continuous=2)
    #     model = MResNet(num_categorical=8, num_continuous=2)

    model.load_state_dict(torch.load(path))

    model.to(DEVICE)
    model.eval()

    return model


def generate_gif(images, output):
    img, *imgs = [Image.fromarray(i, 'RGB') for i in images]
    img.save(fp=output, format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)


def eval_model(model: torch.nn.Module, n_envs: int, iterations: int, reset_step: bool):
    env = gym.make('MineRLTreechop-v0')

    for i in range(n_envs):
        print(f'Execution {i}')
        run_env(model, env, iterations, reset_step)


def run_env(model, env, iterations: int, reset_step: bool, eps=1.0):
    obs = env.reset()
    done = False
    hc = None

    sequence_frames = []
    all_frames = []

    camera_std = []

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

            camera_std.append((pred[0][2].item(), pred[0][3].item()))

            if eps == 1.0:
                action = tensor_to_action_dict(env, pred.squeeze())
            else:
                if int(bernoulli.sample().item()):
                    action = tensor_to_action_dict(env, pred.squeeze())
                else:
                    action, _ = tensor_to_probabilistic_action_dict(env, pred.squeeze())

            obs, rew, done, _ = env.step(action)
            rewards.append(rew)

            idx += 1

    if sum(rewards) > 15:
        global SAVED_GIF_COUNT
        SAVED_GIF_COUNT += 1
        generate_gif(all_frames, os.path.join('results', 'gifs', f'{SAVED_GIF_COUNT}.gif'))

    return camera_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['CNN', 'LSTM'])
    parser.add_argument('--reset_step', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1000)
    args = parser.parse_args()

    load_dotenv()
    eval_model(load_model(args.model), n_envs=args.episodes, iterations=args.iterations, reset_step=args.reset_step)

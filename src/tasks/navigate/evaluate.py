import argparse
import os

import gym
import torch
from dotenv import load_dotenv

from src.models.imitation import PolicyCNNModel, PolicyLSTMModel
from src.tasks.treechop.imitation_train import transformation
from src.utils import tensor_to_action_dict, DEVICE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['CNN', 'LSTM'])
    args = parser.parse_args()

    load_dotenv()

    if args.model == 'CNN':
        model = PolicyCNNModel(out_features=11, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], 'ImitationCNNModel_X.pt')
    else:
        model = PolicyLSTMModel(out_features=11, num_continuous=2)
        path = os.path.join(os.environ['CHECKPOINT_DIR'], 'ImitationLSTMModel_2.pt')

    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()

    env = gym.make('MineRLNavigateDense-v0')

    obs = env.reset()
    done = False

    frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)
    frames = torch.cat(10 * [frame]).unsqueeze_(0)

    with torch.no_grad():
        while not done:
            frame = transformation(obs['pov'].copy()).unsqueeze_(0).to(DEVICE)

            if args.model == 'CNN':
                pred = model(frame).squeeze()
            else:
                frames[0, :-1, :, :, :] = frames[0, 1:, :, :, :].clone()
                frames[0, -1, :, :, :] = frame
                pred = model(frames).squeeze()

            action = tensor_to_action_dict(env, pred)

            print(action)

            obs, rew, done, _ = env.step(action)

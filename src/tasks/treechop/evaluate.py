import os

import gym
import minerl
import torch
from dotenv import load_dotenv

from src.tasks.treechop.imitation_train import transformation
from src.utils import tensor_to_action_dict, DEVICE

load_dotenv()

minerl.data.download(os.environ['DATASET_DIR'], experiment='MineRLNavigate-v0')
data = minerl.data.make('MineRLNavigate-v0', data_dir=os.environ['DATASET_DIR'])

model = torch.load(os.path.join(os.environ['CHECKPOINT_DIR'], '42.pt'))
model.to(DEVICE)
model.eval()

env = gym.make('MineRLNavigate-v0')

obs = env.reset()
done = False

with torch.no_grad():
    while not done:
        pov = torch.stack([transformation(obs['pov'].copy())])
        action = tensor_to_action_dict(model(pov))
        obs, rew, done, _ = env.step(action)

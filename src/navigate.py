import minerl
import os

import torch
import torch.nn as nn
from dotenv import load_dotenv

from src.models.rnn import ImitationModel

load_dotenv()

minerl.data.download(os.environ['DATASET_DIR'], experiment='MineRLNavigate-v0')
data = minerl.data.make('MineRLNavigate-v0', data_dir=os.environ['DATASET_DIR'])

model = ImitationModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=32):
    print(current_state['pov'].shape)

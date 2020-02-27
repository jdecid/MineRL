import os

import minerl
import torch
from dotenv import load_dotenv
from torchvision.transforms import Compose

from src.models.rnn import ImitationRNNModel
from src.utils import ToTensor, ImitationLoss

load_dotenv()

minerl.data.download(os.environ['DATASET_DIR'], experiment='MineRLNavigate-v0')
data = minerl.data.make('MineRLNavigate-v0', data_dir=os.environ['DATASET_DIR'])

model = ImitationRNNModel(out_features=11, num_continuous=2)
optimizer = torch.optim.Adam(model.parameters())
criterion = ImitationLoss(num_continuous=2)

transformation = Compose([ToTensor()])  # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

EPOCHS = 1
SEQ_LEN = 32

for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=EPOCHS, max_sequence_len=SEQ_LEN):
    optimizer.zero_grad()

    pov = current_state['pov']
    pov = torch.stack(list(map(transformation, pov)))

    prediction = model(pov)

    loss = criterion(prediction.squeeze(), action)
    loss.backward()

    optimizer.step()

import os

import minerl
import numpy as np
import torch
from dotenv import load_dotenv
from torchvision.transforms import Compose
from tqdm import tqdm

from src.models.imitation import ImitationCNNModel
from src.utils import ToTensor, ImitationLoss, DEVICE

EPOCHS = 1
SEQ_LEN = 1
BATCH_SIZE = 16

transformation = Compose([ToTensor()])


def next_seq_frame_batch(data, epochs: int, seq_len: int, batch_size: int):
    current_batch_tensors = []
    actions = []

    for current_state, action, _, next_state, _ in data.sarsd_iter(num_epochs=epochs, max_sequence_len=seq_len):
        current_povs = current_state['pov']
        next_povs = next_state['pov']

        povs = np.concatenate((current_povs, next_povs), axis=0)
        povs = torch.stack(list(map(transformation, povs)))

        for i in range(seq_len):
            current_batch_tensors.append(povs[i:seq_len + i])

            # Is this properly?
            actions.append(action)

            if len(current_batch_tensors) == batch_size:
                yield torch.stack(current_batch_tensors), actions
                current_batch_tensors = []
                actions = []

    yield torch.stack(current_batch_tensors), actions


def train(model, data):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = ImitationLoss(2)

    for data, labels in tqdm(next_seq_frame_batch(data, EPOCHS, SEQ_LEN, BATCH_SIZE)):
        if SEQ_LEN == 1:
            data = data.squeeze()
        data = data.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        prediction = model(data).squeeze()

        loss = criterion(prediction, labels)
        loss.backward()

        print(loss)

        optimizer.step()

    torch.save(model, os.path.join(os.environ['CHECKPOINT_DIR'], '42.pt'))


if __name__ == '__main__':
    def main():
        load_dotenv()

        minerl.data.download(os.environ['DATASET_DIR'], experiment='MineRLTreechop-v0')
        data = minerl.data.make('MineRLTreechop-v0', data_dir=os.environ['DATASET_DIR'])
        print('Data Loaded')

        model = ImitationCNNModel(out_features=11, num_continuous=2)
        # model = ImitationRNNModel(out_features=11, num_continuous=2)
        model = model.to(DEVICE)

        train(model, data)


    main()

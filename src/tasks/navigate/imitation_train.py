import argparse
import os

import minerl
import torch
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from src.models.imitation import ImitationLSTMModel, ImitationCNNModel
from src.utils import ToTensor, ImitationLoss, DEVICE

EPOCHS = 100
SEQ_LEN = 10
BATCH_SIZE = 64

transformation = Compose([ToTensor()])


def next_batch(data, seq_len: int, epochs: int, batch_size: int):
    batch_frames = []
    batch_actions = []

    for current_state, action, _, _, _ in data.sarsd_iter(num_epochs=epochs, max_sequence_len=seq_len):
        frames = current_state['pov']
        frames = torch.stack(list(map(transformation, frames)))

        # Force last possible remaining smaller size batch to have the same size than the others by repeating the
        # last element. By doing this, we can stack it with the others without problems.
        # while batch_frames[-1] == 0:
        #    batch_frames.append(batch_frames[-1])
        #    batch_actions.append(batch_actions[-1])
        if frames.size(0) == seq_len:
            batch_frames.append(frames)
            batch_actions.append(action)

        if len(batch_frames) == batch_size:
            batch_frames = torch.stack(batch_frames)
            yield batch_frames if seq_len > 1 else batch_frames.squeeze(), batch_actions

            batch_frames = []
            batch_actions = []

    batch_frames = torch.stack(batch_frames)
    yield batch_frames if seq_len > 1 else batch_frames.squeeze(), batch_actions


def train(model, frames):
    writer = SummaryWriter(log_dir=os.environ['LOGS_DIR'])

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = ImitationLoss(num_continuous=2, writer=writer)

    seq_len = SEQ_LEN if isinstance(model, ImitationLSTMModel) else 1
    for idx, (frames, target_actions) in enumerate(next_batch(frames, seq_len, EPOCHS, BATCH_SIZE), start=1):
        frames = frames.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        prediction = model(frames).squeeze()

        loss = criterion(prediction, target_actions, idx)
        loss.backward()

        optimizer.step()

        if idx % 10000 == 0:
            print(f'Saving checkpoint {idx // 10000}')
            torch.save(model.state_dict(),
                       os.path.join(os.environ['CHECKPOINT_DIR'], f'{model.__class__.__name__}_{idx // 10000}.pt'))


def main(args: argparse.Namespace):
    load_dotenv()

    minerl.data.download(os.environ['DATASET_DIR'], experiment='MineRLNavigateDense-v0')
    data = minerl.data.make('MineRLNavigateDense-v0', data_dir=os.environ['DATASET_DIR'])
    print('Data Loaded')

    if args.model == 'CNN':
        model = ImitationCNNModel(out_features=11, num_continuous=2)
    else:  # LSTM
        model = ImitationLSTMModel(out_features=11, num_continuous=2)
    model = model.to(DEVICE)

    train(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=['CNN', 'LSTM'])
    main(parser.parse_args())

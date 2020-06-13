import argparse
from datetime import datetime

import torch
from dotenv import load_dotenv

from src.models.imitation import PolicyCNNModel, PolicyLSTMModel
from src.tasks.treechop.evaluate import load_model
from src.utils import DEVICE


def main(training_type: str, model_type: str, checkpoint: str, n_envs: int):
    if checkpoint is not None:
        model = load_model(model_type)
    else:
        if model_type == 'CNN':
            model = PolicyCNNModel(num_categorical=8, num_continuous=2)
        else:  # LSTM
            model = PolicyLSTMModel(num_categorical=8, num_continuous=2)
    model = model.to(DEVICE)

    run_timestamp = str(datetime.now())
    run_timestamp = run_timestamp.replace(':', '-').replace(' ', '_').replace('.', '-')

    if training_type == 'Imitation':
        from src.tasks.treechop.imitation_train import main as train
        train(model, run_timestamp)
    else:  # Reinforcement Learning
        from src.tasks.treechop.reinforcement_cnn_train import main as train
        train(model, n_envs, run_timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=str, choices=['Imitation', 'RL'])
    parser.add_argument('model', type=str, choices=['CNN', 'LSTM'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--nenvs', type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    with torch.autograd.set_detect_anomaly(True):
        main(args.training, args.model, args.checkpoint, args.nenvs)

import argparse
from datetime import datetime

from src.models.imitation import PolicyCNNModel, PolicyLSTMModel
from src.utils import DEVICE


def main(training_type: str, model_type: str, n_envs: int):
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
        from src.tasks.treechop.reinforcement_train import main as train
        train(model, n_envs, run_timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=str, choices=['Imitation', 'RL'])
    parser.add_argument('model', type=str, choices=['CNN', 'LSTM'])
    parser.add_argument('--nenvs', type=int, default=1)
    args = parser.parse_args()

    main(args.training, args.model, args.nenvs)

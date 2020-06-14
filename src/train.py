import argparse
from datetime import datetime

from dotenv import load_dotenv

from src.models.ac import ValueCNNModel
from src.models.imitation import PolicyCNNModel, PolicyLSTMModel
from src.tasks.treechop.evaluate import load_model
from src.utils import DEVICE


def main(args):
    training_type = args.training
    model_type = args.model
    checkpoint = args.checkpoint
    episodes = args.episodes
    iterations = args.iterations
    eps = args.eps
    reg = args.reg
    reset_step = args.reset_step

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
    elif training_type == 'REINFORCE':
        from src.tasks.treechop.reinforcement_train import main as train
        train(model, episodes, iterations, eps, reset_step, run_timestamp)
    elif training_type == 'AC':
        from src.tasks.treechop.ac_train import main as train
        # value_model = ValueCNNModel(load_model('CNN')).to(DEVICE)
        value_model = load_model('Value')
        train(model, value_model, episodes, iterations, eps, reset_step, reg, run_timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=str, choices=['Imitation', 'REINFORCE', 'AC'])
    parser.add_argument('model', type=str, choices=['CNN', 'LSTM'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--eps', type=float, default=0.9)
    parser.add_argument('--reg', type=float, default=0)
    parser.add_argument('--reset_step', type=bool, default=False, const=True, nargs='?')
    args = parser.parse_args()

    load_dotenv()
    main(args)

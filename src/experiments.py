import os

import gym
import numpy as np
from dotenv import load_dotenv

from src.tasks.treechop.evaluate import run_env, load_model

file_path = os.path.join('results', 'experiments', 'results_9.txt')


def eval(env, model, episodes, iterations, reset_step, eps=1.0):
    results = []
    for ep in range(episodes):
        print(f'Running episode {ep + 1}')
        rewards = run_env(model, env, iterations, reset_step, eps)
        results += rewards
        # rewards = np.array(list(map(float, rewards)))
        # results.append(rewards.sum())

    zipped = list(zip(*results))
    print(len(zipped))
    print(np.mean(zipped[0]))
    print(np.mean(zipped[1]))

    return results


def store_results(description, results):
    with open(file_path, mode='a') as f:
        f.write(description)
        f.write(','.join(list(map(str, results))))
    print('Stored ' + description)


def main():
    open(file_path, mode='w').close()

    env = gym.make('MineRLTreechop-v0')

    description = '[Imitation learning CNN achieves good results straightforward]\n'
    model = load_model('AC_CNN')
    results = eval(env, model, episodes=5, iterations=1000, reset_step=False, eps=0.4)
    store_results(description, results)
    #
    # description = '[Imitation learning CNN+LSTM low features + dropout good results (window)]\n'
    # model = load_model('LSTM')
    # results = eval(env, model, episodes=100, iterations=1000, reset_step=True)
    # store_results(description, results)
    #
    # description = '[Imitation learning CNN+LSTM low features + dropout good results (infinite)]\n'
    # model = load_model('LSTM')
    # results = eval(env, model, episodes=100, iterations=1000, reset_step=False)
    # store_results(description, results)
    #
    # description = '[Imitation learning CNN + stochastic eps 0.8]\n'
    # model = load_model('CNN')
    # results = eval(env, model, episodes=100, iterations=1000, reset_step=False, eps=0.8)
    # store_results(description, results)
    #
    # description = '[Imitation learning CNN + stochastic eps 0.4]\n'
    # model = load_model('CNN')
    # results = eval(env, model, episodes=100, iterations=1000, reset_step=False, eps=0.4)
    # store_results(description, results)
    #
    # description = '[Reinforce CNN + stochastic eps 0.4]\n'
    # model = load_model('CNN')
    # results = eval(env, model, episodes=100, iterations=1000, reset_step=False, eps=0)
    # store_results(description, results)

    # TODO: REINFORCE LSTM infinite

    # description = '[AC CNN + stochastic eps 0.4]\n'
    # model = load_model('AC_CNN')
    # results = eval(env, model, episodes=80, iterations=1000, reset_step=False, eps=0.4)
    # store_results(description, results)

    # description = '[ResNet CNN + stochastic eps 0.4]\n'
    # model = load_model('LSTM')
    # results = eval(env, model, episodes=100, iterations=1000, reset_step=False)
    # store_results(description, results)

    # TODO: AC CNN value bias


if __name__ == '__main__':
    load_dotenv()
    main()

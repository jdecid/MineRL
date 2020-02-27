import gym
import minerl

# Run a random agent through the environment
env = gym.make("MineRLNavigate-v0")  # A MineRLNavigateDense-v0 env

obs = env.reset()
done = False

while not done:
    action = {'forward': 1, 'left': 1, 'attack': 1, 'jump': 1}
    obs, rew, done, _ = env.step(action)
    # Do something

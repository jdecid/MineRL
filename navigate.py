import gym
import minerl

# Run a random agent through the environment
env = gym.make("MineRLNavigate-v0")  # A MineRLNavigate-v0 env

obs = env.reset()
done = False

while not done:
    # Take a no-op through the environment.
    obs, rew, done, _ = env.step(env.action_space.noop())
    # Do something

######################################

# Sample some data from the dataset!
data = minerl.data.make("MineRLNavigate-v0")

# Iterate through a single epoch using sequences of at most 32 steps
for obs, rew, done, act in data.seq_iter(num_epochs=1, batch_size=32):
    pass

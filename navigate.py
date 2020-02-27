import gym
import minerl

# Run a random agent through the environment
env = gym.make("MineRLNavigate-v0")  # A MineRLNavigate-v0 env

obs = env.reset()
done = False

print(env.action_space)

while not done:
    # Take a no-op through the environment.
    obs, rew, done, _ = env.step(
        {
            'attack': 1,
            'back': 0,
            'camera': (0, 0),
            'forward': 1,
            'jump': 1,
            'left': 0,
            'place': 'none',
            'right': 0,
            'sneak': 0,
            'sprint': 0
        }
    )
    # Do something

######################################

# Sample some data from the dataset!
data = minerl.data.make("MineRLNavigate-v0")

# Iterate through a single epoch using sequences of at most 32 steps
for obs, rew, done, act in data.seq_iter(num_epochs=1, batch_size=32):
    pass

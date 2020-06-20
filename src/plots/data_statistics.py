import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

paths = glob(os.path.join('dataset', 'MineRLTreechop-v0', '*'))

num_frames = []
num_timesteps = []
num_recordings = len(paths)

for path in paths:
    with open(os.path.join(path, 'metadata.json')) as f:
        data = '\n'.join(f.readlines())
        data = json.loads(data)
        num_frames.append(data['duration_steps'])
        num_timesteps.append(data['duration_ms'])

print(f'Recordings: {num_recordings}')
print(f'#Frames: {sum(num_frames)}')
print(f'Gameplay time: {sum(num_timesteps)}ms ({sum(num_timesteps) / 3.6e6:.4f}h)')

plt.title('Frames per expert data replay', fontsize=16)
plt.xlabel('#Frames', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.hist(num_frames, histtype='bar', ec='black')
plt.savefig('results/plots/frames_per_replay.png')
plt.show()

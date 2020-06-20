import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator

base_event_path = 'CNN_BASE_IMITATION'
scalar_event_path = 'Loss_MSE Loss'
event_path = 'events.out.tfevents.1591650349.DESKTOP-8ORT4I6.20336.'

path = os.path.join('logs', base_event_path, scalar_event_path, event_path)

ea = event_accumulator.EventAccumulator(path + '2',
                                        size_guidance={
                                            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                            event_accumulator.IMAGES: 4,
                                            event_accumulator.AUDIO: 4,
                                            event_accumulator.SCALARS: 0,
                                            event_accumulator.HISTOGRAMS: 1,
                                        })

ea.Reload()

loss = []
for e in ea.Scalars('Loss'):
    loss.append(e.value)

df = pd.DataFrame(np.array(loss))

plt.plot(loss, 'lightblue', df[0].rolling(10).mean(), 'blue')
plt.show()

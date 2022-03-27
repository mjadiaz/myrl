import os
import pickle
import pandas as pd

path = '/scratch/mjad1g20/PhenoGame/myrl/runs/chckpt_50'
with open(os.path.join(path, 'memory.pickle'), 'rb') as f:
        memory = pickle.load(f)

print(len(memory))

progress = pd.read_csv(os.path.join(path, 'progress.csv'))
print(progress)

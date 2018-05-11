import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from trackml.dataset import load_event
from trackml.score import score_event
import pdb
import pandas as pd
import csv

### load data ###
hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')

true_tracks = np.load("../port_toy/all_tracks.npy")

print(truth.head())
print(cells.head())

if not os.path.exists("bruyant.npy"):
    noise = []
    x = []
    y = []
    z = []
    q = []
    for i in range(len(truth)):
        if truth.iloc[i][1] == 0:
            noise.append(truth.iloc[i])
            x.append(truth.iloc[i][2])
            y.append(truth.iloc[i][3])
            z.append(truth.iloc[i][4])
            for j in range(len(cells)):
                if cells.iloc[j][0] == truth.iloc[i][0]:
                    q.append(cells.iloc[j][3])
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    q = np.asarray(q)
    
    A = [x, y, z, q]
    A = np.asarray(A)
    np.save("bruyant.npy", A)

else:
    A = np.load("bruyant.npy")
    x = A[0]
    y = A[1]
    z = A[2]
    q = A[3]



plt.hist2d((x**.5 + y**.5), q)
plt.show()


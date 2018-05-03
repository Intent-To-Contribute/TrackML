import tensorflow as tensorflow
import os, sys
import csv
import pandas as pd
import numpy as np
from trackml.dataset import load_event

hits, cells, particles, truth = load_event('../..//Data/train_100_events/event000001052')
sorted_truth = truth.sort_values("particle_id")

print(hits.iloc[10][0], "\t", hits.iloc[10][1], "\t", hits.iloc[10][2], "\t", hits.iloc[10][3])

file = open("SeedCandidates.txt", "w+")
for i in range(len(hits)):
    if abs(hits.iloc[i][1]) < .1:
        file.write(str(hits.iloc[i][0]))
        file.write("\n")
file.close()




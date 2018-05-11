import sys
from trackml.dataset import load_event
import numpy as np
import pdb
from tempfile import TemporaryFile


# -- reading in data --
hits, cells, particles, truth = load_event('../..//Data/train_100_events/event000001052')
sorted_truth = truth.sort_values("particle_id")

example_hits = []
for i in range(len(sorted_truth)):
    if i > 19409 and i < 19416:
        example_hits.append(sorted_truth.iloc[i][0])


vid = []
lid = []
mid = []

for hit in example_hits:
    print(hits.loc[hits['hit_id'] == hit][0])


# endomundo

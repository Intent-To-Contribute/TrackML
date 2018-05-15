# import os
# import sys
import numpy as np
# import matplotlib.pyplot as plt
from trackml.dataset import load_event
# from trackml.score import score_event
# import pdb
# import pandas as pd
# import csv

from data_formatter import DataFormatter

## Load Data ##
path_to_dataset = "../../Data/train_100_events/"
event_path = "event000001052"
model_name = "identity.keras"

hits, cells, particles, truth = load_event(path_to_dataset + event_path)

# Get the sorted tracks
formatter = DataFormatter()
true_tracks, hit_tracks = formatter.getSortedTracks(particles, truth, hits)

## Load Predicted Seeds ##
seed_file = open("SeedCandidates.txt", "r")
our_tracks = []
seed_hits = []
np_hits = np.asarray(hits)
for seed_id in seed_file:
    seed_id = int(float(seed_id.strip()))
    seed_hit = np_hits[np_hits[:,0] == seed_id][0]
    our_tracks.append([int(seed_hit[0])])
    seed_hits.append(seed_hit)

print("\nStarting with " + str(len(seed_hits)) + " seed hits")

## Evaluate Predicted Seeds ##
true_seed_ids = []
for track in true_tracks:
    true_seed_ids.append(track[0][0])

seed_ids = []
for seed_hit in seed_hits:
    seed_ids.append(seed_hit[0])

found_seeds = np.isin(seed_ids, true_seed_ids)
num_seeds_found = np.count_nonzero(found_seeds)
num_seeds_guessed = len(seed_hits)
num_real_seeds = len(true_seed_ids)

print(num_seeds_found, "/", num_real_seeds, " seeds found with", num_seeds_guessed, "predicted.\n")
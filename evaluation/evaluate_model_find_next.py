# import os
import sys, math
import numpy as np
# import matplotlib.pyplot as plt
from trackml.dataset import load_event
from trackml.score import score_event
# import pdb
# import pandas as pd
# import csv

# from utils import find_nearest_hit, scan_voxels_for_hits, sort_tracks
from data_formatter import DataFormatter
from voxels import Voxels

import tensorflow as tensorflow
import keras
from keras.models import load_model
print("\n", end="")

## Load Data ##
path_to_dataset = "../../Data/train_100_events/"
event_path = "event000001052"
model_name = "identity.keras"

hits, cells, particles, truth = load_event(path_to_dataset + event_path)
# true_tracks = np.load("../port_toy/all_tracks.npy")

# TODO do the appropriate transform on truth["x", "y", "z"] and hits["x", "y", "z"]


# Get the sorted tracks
formatter = DataFormatter()
true_tracks, hit_tracks, max_len = formatter.getSortedTracks(particles, truth, hits)
print("Max length:", max_len)


## Create Voxels ##
hit_voxels = Voxels(hits, 200)

## Import Model ##
model = load_model(model_name)
# print(model.summary())

## Evaluate Predictions ##
for i in range(3, max_len):
    x, y = formatter.getInputOutput(true_tracks, i, i)
    print("\nBatch size for hit #%d:" % (i+1), x.shape[0])
    
    if (len(x) == 0): break
    print("Find predictions... ", end="")
    predicted = model.predict(x)
    print("done.")
    # print("Finished predicting hits")

    print("Find closest hits... ", end="")
    predicted_hits = []
    total = predicted.shape[0]
    percent = math.ceil(total / 100)
    for idx, guess in enumerate(predicted):
        if idx % percent == 0: print("\rFind closest hits... " + str(int(100*idx / total)) + "%", end="")
        hit = hit_voxels.findClosestPoint(*guess)
        predicted_hits.append(hit)
    print("\rFind closest hits... 100% complete")
    predicted_hits = np.asarray(predicted_hits)


    true_ids = []
    for hit in y:
        true_ids.append(hit[0])
    true_ids = np.asarray(true_ids)

    predicted_ids = []
    for prediction in predicted_hits:
        predicted_ids.append(prediction[0])
    predicted_ids = np.asarray(predicted_ids)
    print("Unique ids predicted:", np.unique(predicted_ids).shape[0])

    found = np.equal(predicted_ids, true_ids)
    num_true_found = np.count_nonzero(found)
    num_true = len(true_ids)
    print("%.4f%% correctly predicted for hit #%d" % (100*num_true_found / num_true, i+1))


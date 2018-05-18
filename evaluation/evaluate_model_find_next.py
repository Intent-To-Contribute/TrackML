import sys, os, math
import numpy as np
from trackml.dataset import load_event
from trackml.score import score_event

from data_formatter import DataFormatter
from voxels import Voxels

import tensorflow as tensorflow
import keras
from keras.models import load_model
print("\n", end="")

## Load Data ##
path_to_dataset = "../../Data/train_100_events/"
event_path = "event000001052"
# model_name = "identity.keras"
model_name = "3in_3out.keras"

hits, cells, particles, truth = load_event(path_to_dataset + event_path)
# true_tracks = np.load("../port_toy/all_tracks.npy")

# TODO use the appropriate transform on truth["x", "y", "z"] and hits["x", "y", "z"]


# Get the sorted tracks
formatter = DataFormatter()
true_tracks, hit_tracks, max_len = formatter.getSortedTracks(particles, truth, hits)
print("Max length of a track:", max_len)


## Create Voxels ##
hit_voxels = Voxels(hits, 200)

## Import Model ##
model = load_model(model_name)

## Evaluate Predictions ##
for i in range(3, max_len-1):
    x, y = formatter.getInputOutput(true_tracks, i, i, 18)
    print("\nBatch size for predicting hit #%d:" % (i+1), x.shape[0])
    
    if (len(x) == 0): break
    print("Find predictions... ", end="")
    predicted = model.predict(x)
    print("done.")

    print("Find closest hits... ", end="")
    predicted_hits = []
    total = predicted.shape[0]
    percent = math.ceil(total / 100)
    for idx, guess in enumerate(predicted):
        if idx % percent == 0: print("\rFind closest hits... " + str(int(100*idx / total)) + "%", end="")
        hit = hit_voxels.findClosestPoint(*guess)
        predicted_hits.append(hit)
    predicted_hits = np.asarray(predicted_hits)
    print("\rFind closest hits... 100% complete")

    true_ids = y[:,0]
    predicted_ids = predicted_hits[:,0]
    print("Unique ids predicted:", np.unique(predicted_ids).shape[0])
    found = np.equal(predicted_ids, true_ids)
    num_true_found = np.count_nonzero(found)
    num_true = len(true_ids)
    print("%.4f%% correctly predicted for hit #%d" % (100*num_true_found / num_true, i+1))


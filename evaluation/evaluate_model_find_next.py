from utils import find_nearest_hit, scan_voxels_for_hits, sort_tracks
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from trackml.dataset import load_event
from trackml.score import score_event
import pdb
import pandas as pd
import csv

## Load Data ##
path_to_dataset = "../../Data/train_100_events/"
event_path = "event000001052"

hits, cells, particles, truth = load_event(path_to_dataset + event_path)
true_tracks = np.load("../port_toy/all_tracks.npy")

## Create Voxels ##
# find min/max of x,y,z
xMax = -sys.maxsize
yMax = -sys.maxsize
zMax = -sys.maxsize
xMin = sys.maxsize
yMin = sys.maxsize
zMin = sys.maxsize
for track in true_tracks:
    for hit in track:
        if (xMax < hit[2]): xMax = hit[2]
        if (yMax < hit[3]): yMax = hit[3]
        if (zMax < hit[4]): zMax = hit[4]
        if (xMin > hit[2]): xMin = hit[2]
        if (yMin > hit[3]): yMin = hit[3]
        if (zMin > hit[4]): zMin = hit[4]

hits = np.asarray(hits)
xRange = xMax - xMin
yRange = yMax - yMin
zRange = zMax - zMin
n = 150
voxels = np.zeros((n+1,n+1,n+1), dtype=object)

for hit in hits:
    xHit = hit[1]
    yHit = hit[2]
    zHit = hit[3]
    i = int(n * ((xHit - xMin) / xRange))
    j = int(n * ((yHit - yMin) / yRange))
    k = int(n * ((zHit - zMin) / zRange))
    if voxels[i][j][k] == 0:
        voxels[i][j][k] = []
    voxels[i][j][k].append(hit)

print("finished creating voxels")

## Load Predicted Seeds ##
seed_file = open("SeedCandidates.txt", "r")
our_tracks = []
seed_hits = []
for seed_id in seed_file:
    seed_id = int(float(seed_id.strip()))
    seed_hit = hits[hits[:,0] == seed_id][0]
    our_tracks.append([int(seed_hit[0])])
    seed_hits.append(seed_hit)

print()
print("starting with " + str(len(seed_hits)) + " seed hits")

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

print(num_seeds_found, "/", num_real_seeds, " seeds found with", num_seeds_guessed, "predicted.")
print("recall", num_seeds_found / num_real_seeds)
print("precision", num_seeds_found / num_seeds_guessed)
print()

## Import Model ##
import tensorflow as tensorflow
import keras
from keras.models import load_model

model = load_model("3in_3out.keras")
print(model.summary())

## Evaluate Predictions ##
max_len = 18
for i in range(1, max_len):
    # print("Evaluating Hit #%d..." % (i+1))
    x = []
    true_hits = []
    for track in true_tracks:
        x_hit = np.zeros((max_len, 3))

        if i < len(track)-1:
            for z in range(i):
                x_hit[max_len-i+z] = track[z][2:5]
            x.append(x_hit)
            true_hits.append(track[i])
    
    if (len(x) == 0): break
    x = np.asarray(x)

    y = model.predict(x)
    # print("Finished predicting hits")

    predicted_hits = []
    counter = 0
    for guess in y:    
        xHit = guess[0]
        yHit = guess[1]
        zHit = guess[2]
        ii = int(n * ((xHit - xMin) / xRange))
        j = int(n * ((yHit - yMin) / yRange))
        k = int(n * ((zHit - zMin) / zRange))
        
        possible_nearest_hits = scan_voxels_for_hits(voxels, n, ii, j, k)
        hit = find_nearest_hit(possible_nearest_hits, guess)
        predicted_hits.append(hit)
        # if (counter % 5000) == 0:
        #     print(str(counter) + "/" + str(len(y)))
        #     print(possible_nearest_hits.shape)
        counter += 1

    # print("Finished finding closest hits to predictions")
    predicted_hits = np.asarray(predicted_hits)
    # print(predicted_hits.shape)

    true_ids = []
    for hit in true_hits:
        true_ids.append(hit[0])
    true_ids = np.asarray(true_ids)

    predicted_ids = []
    for hit in predicted_hits:
        predicted_ids.append(hit[0])
    predicted_ids = np.asarray(predicted_ids)

    found = np.equal(predicted_ids, true_ids)

    num_true_found = np.count_nonzero(found)
    num_true = len(true_ids)

    print("%.4f%% accuracy for hit #%d" % (100*num_true_found / num_true, i+1))


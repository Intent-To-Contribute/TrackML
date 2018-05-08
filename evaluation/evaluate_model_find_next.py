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

### load data ###
hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')

true_tracks = np.load("../port_toy/all_tracks.npy")

# for event_id, hits, cells, particles, truth in load_dataset('path/to/dataset'):
#score = score_event(truth, shuffled)



## find min/max of x,y,z ##
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

## creating voxels ##
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


### seeds ###
seed_file = open("SeedCandidates.txt", "r")
our_tracks = []
seed_hits = []
for seed_id in seed_file:
    seed_id = int(float(seed_id.strip()))
    seed_hit = hits[hits[:,0] == seed_id][0]
    our_tracks.append([int(seed_hit[0])])
    seed_hits.append(seed_hit)

print("starting with " + str(len(seed_hits)) + " seed hits")

## evaluate seed finding ##
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


## build input vectors ##
x = []
for seed_hit in seed_hits:
	input_vector = np.zeros((18, 3))
	input_vector[17] = seed_hit[1:4]
	x.append(input_vector)

## predict the next point ##
import tensorflow as tensorflow
import keras
from keras.models import load_model

## Import the stopping model
stopping_model = load_model("final_classifier.keras")
    
x = np.asarray(x)
model = load_model("3in_3out.keras")
print(model.summary())
if not os.path.exists("guesses_2.npy"):
    y = model.predict(x)
    print("finished predicting next hits")
    np.save("guesses_2.npy", y)
else:
    y = np.load("guesses_2.npy")


## for each prediction, find the closest hit to it ##
next_hits = []
counter = 0
for guess in y:    
    xHit = guess[0]
    yHit = guess[1]
    zHit = guess[2]
    i = int(n * ((xHit - xMin) / xRange))
    j = int(n * ((yHit - yMin) / yRange))
    k = int(n * ((zHit - zMin) / zRange))
    
    possible_nearest_hits = scan_voxels_for_hits(voxels, n, i, j, k)
    next_hit = find_nearest_hit(possible_nearest_hits, guess)
    next_hits.append(next_hit)
    if (counter % 1000) == 0:
        print(str(counter) + "/" + str(len(y)))
        print(possible_nearest_hits.shape)
    counter += 1

print("finished finding closest hits to predictions")


# for i in range(17):    
# #for i in range(1):    
#     new_x = []
#     for j in range(len(x)):
#         input_vector = np.zeros((18, 3))
#         for k in range(1, 17):
#             input_vector[k] = x[j][k+1]
#         # our_tracks[j].append(int(next_hits[j][0]))
#         input_vector[17] = next_hits[j][1:4]
#         new_x.append(input_vector)

#     ## decide if this is the end or not ##
#     stop_predictions = stopping_model.predict(np.asarray(new_x))
#     stop_predictions = np.argmax(stop_predictions, axis=1)
#     print(np.count_nonzero(stop_predictions))
    
#     ## if predicted to have stopped, then add to our_tracks
    
#     ## if predicted to not have stopped, then continue along the pipeline 
#     new_x = np.asarray(new_x)
#     print("value of 1", stop_predictions[stop_predictions[:] == 1])
#     print("value of 0", stop_predictions[stop_predictions[:] == 0])
#     print("shapes", np.asarray(new_x).shape, stop_predictions.shape)
#     print("continue with these sequences", new_x[stop_predictions[:] == 1])
#     print("these sequences have sToPPeD", new_x[stop_predictions[:] == 0])
#     finished_tracks = new_x[stop_predictions[:] == 0]
#     pdb.set_trace()

#     our_tracks = np.concatenate(np.asarray(our_tracks), finished_tracks)
#     new_x = new_x[stop_predictions[:] == 1]

#     print("-" * 35)
#     print("our_tracks", our_tracks, our_tracks.shape)
#     print("new_x", new_x, new_x.shape)
#     print("-" * 35)

#     ## predict the next point ##
#     print("predicting next hits " + str(i))
#     x = new_x
#     x = np.asarray(x)
#     y = model.predict(x)
#     print("finished predicting next hits " + str(i))


#     ## for each prediction, find the closest hit to it ##
#     print("finding closest hits to predictions " + str(i))
#     next_hits = []
#     counter = 0
#     for guess in y:    
#         xHit = guess[0]
#         yHit = guess[1]
#         zHit = guess[2]
#         ii = int(n * ((xHit - xMin) / xRange))
#         j = int(n * ((yHit - yMin) / yRange))
#         k = int(n * ((zHit - zMin) / zRange))
        
#         possible_nearest_hits = scan_voxels_for_hits(voxels, n, ii, j, k)
#         next_hit = find_nearest_hit(possible_nearest_hits, guess)
#         next_hits.append(next_hit)
#         if (counter % 1000) == 0:
#             print(str(counter) + "/" + str(len(y)))
#             print(possible_nearest_hits.shape)
#         counter += 1

#     print("finished finding closest hits to predictions " + str(i))


# ## format data into tracks ##

# print(our_tracks)
# submission = []
# for i in range(len(our_tracks)):
#     for hit in our_tracks[i]:
#         submission.append([hit, i])

# print(submission)
# np.save("submission_1052.npy", np.asarray(submission))
# df = pd.DataFrame(np.asarray(submission), columns = ["hit_id", "track_id"])
# print(df.head())


# score = score_event(truth, df)
# print(score)

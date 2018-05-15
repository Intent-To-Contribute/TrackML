import os, sys
import csv
import pandas as pd
import numpy as np
from trackml.dataset import load_event

from sklearn.preprocessing import StandardScaler
import transforms

### load data ###
hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')

# test transformations
#   transforms append new columns to hits, in place
transformations = [
    "dbscan_trans",
    "spherical",
    "cylindrical",
    "normalize",
    "standard",
    "identity"
]

# add the tranformed coordinates as new columns to the hits dataframe
for transform in transformations:
    getattr(transforms, transform)(hits) # equivalent to transforms.transform(hits)

# print("hits", hits)

# create a mapping from column_names to the corresponding index (in 'hits')
column_index = {}
for column_name in hits.head():
    column_index[column_name] = hits.columns.get_loc(column_name)

## mapping from transformation to start, end indices ##
transformation_indices = {
    "dbscan_trans"  : [column_index['1_db'], column_index['3_db']],
    "spherical"     : [column_index['1_sph'], column_index['3_sph']],
    "cylindrical"   : [column_index['1_cyl'], column_index['3_cyl']],
    "normalize"     : [column_index['1_norm'], column_index['4_norm']],
    "standard"      : [column_index['1_ss'], column_index['3_ss']],
    "identity"      : [column_index['1_id'], column_index['3_id']]
}


### group hits into their true tracks
##      true coordinates -> true_tracks and
##      hit coordinates  -> hit_tracks 
print("Initializing true_tracks and hit_tracks")
true_tracks = []
hit_tracks = []
if not os.path.exists("true_tracks.npy") or not os.path.exists("hit_tracks.npy"):
    sorted_truth = truth.sort_values("particle_id")
    np_hits = np.asarray(hits)
    np_particles = np.asarray(particles)
    current_pid = -1
    track = []
    for i in range(len(sorted_truth)):
        truth = np.asarray(sorted_truth.iloc[i])
        if truth[1] == 0: continue
        if truth[1] != current_pid and current_pid != -1:
            current_pid = truth[1]
            if len(track) == 0: continue
            track = np.asarray(track)
            # sort by distance from vertex
            track = track[track[:,9].argsort()]
            true_tracks.append(track)

            # get Hits from hit_id's,
            #   first column in truth file is hit_id
            hitIDs = track[track[:,9].argsort()][:, 0]
            hit_track = []
            for hitID in hitIDs:
                hit_track.append(np_hits[int(hitID)-1])
            hit_tracks.append(hit_track)

            track = []
        else:
            current_pid = truth[1]
            particle = np_particles[np_particles[:, 0] == current_pid]
            particle = np.squeeze(particle)
            
            # Add a "distance from vertex" column to each hit
            #   enables sorting hits within a track
            #   (i.e. 1st hit, 2nd hit, etc.)
            vertex = particle[1:4]
            trueHitPoint = truth[2:5]
            diff = np.subtract(vertex, trueHitPoint)
            dist = np.linalg.norm(diff)
            truth = np.append(truth, [dist])
            
            track.append(truth)
    np.save("true_tracks.npy", true_tracks)
    np.save("hit_tracks.npy", hit_tracks)
else:
    true_tracks = np.load("true_tracks.npy")
    hit_tracks = np.load("hit_tracks.npy")
print("Finished initializing true_tracks and hit_tracks")


### Format input sequences X and output Y ###
# --- making input vectors ---
print()
print("Start making input vectors")
if not os.path.exists("true_X.npy") or not os.path.exists("true_Y.npy"):
    true_X = []
    true_Y = []
    for track in true_tracks:
        for i in range(1, len(track)-1):
            x_hit = np.zeros((0, 3))
            for z in range(i):
                hit_to_add = np.asarray(track[z][2:5]).reshape(1,3)
                x_hit = np.concatenate((x_hit, hit_to_add))
            true_X.append(x_hit)
            true_Y.append(track[i+1][2:5])
    true_X = np.asarray(true_X)
    true_Y = np.asarray(true_Y)
    np.save("true_X.npy", true_X)
    np.save("true_Y.npy", true_Y)
else:
    true_X = np.load("true_X.npy")
    true_Y = np.load("true_Y.npy")

transformed_hits_X = [[] for transform in transformations]
transformed_hits_Y = [[] for transform in transformations]
for hit in hit_tracks:
    for idx, start_end in enumerate(list(transformation_indices.values())):
        start = start_end[0]
        end = start_end[1]+1
        seq_len = end-start
        for i in range(1, len(hit)-1):
            x_hit = np.zeros((0, seq_len))
            for z in range(i):
                hit_to_add = np.asarray(hit[z][start:end]).reshape(1,seq_len)
                x_hit = np.concatenate((x_hit, hit_to_add))
            transformed_hits_X[idx].append(x_hit)
            transformed_hits_Y[idx].append(hit[i+1][start:end])
transformed_hits_X = [np.asarray(transformed_X) for transformed_X in transformed_hits_X]
transformed_hits_Y = [np.asarray(transformed_Y) for transformed_Y in transformed_hits_Y]
print("Finished making input vectors")


import tensorflow as tensorflow
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

# sanity checks for shape
printShapes = False
if (printShapes):
    print()
    print("-"*20 + " random input vector shapes before padding " + "-"*20)
    for i in range(3):
        rand_input = np.random.choice(true_X)
        print("true input", rand_input.shape)

    for transformed_X in transformed_hits_X:
        for i in range(3):
            rand_input = np.random.choice(transformed_X)
            print("transformed input", rand_input.shape)

### Begin Pad sequences ###
true_X = keras.preprocessing.sequence.pad_sequences(true_X, maxlen=None, dtype=np.float64)
for idx, transformed_X in enumerate(transformed_hits_X):
    transformed_hits_X[idx] = keras.preprocessing.sequence.pad_sequences(transformed_X, maxlen=None, dtype=np.float64)
### End Pad sequences ###

# sanity checks for shape
if (printShapes):
    print()
    print("-"*20 + " random input vector shapes after padding " + "-"*20)
    rand_inputs = np.random.randint(true_X.shape[0], size=3)
    for rand_input in rand_inputs:
        print("true input", true_X[rand_input].shape)
        print("true output", true_Y[rand_input].shape)

    for rand_input in rand_inputs:
        for transformed_X in transformed_hits_X:
            print("transformed input", transformed_X[rand_input].shape)
        for transformed_Y in transformed_hits_Y:
            print("transformed output", transformed_Y[rand_input].shape)
            
    print("batch input shape -- true", true_X.shape)
    for transformed_X in transformed_hits_X:
        print("batch input shape -- transformed", transformed_X.shape)
    print()


## Create the LSTM model ##
trainTrue = False
if (trainTrue):
    in_neurons = 3
    out_neurons = 3
    hidden_neurons = 500

    model = Sequential()
    model.add(LSTM(hidden_neurons, return_sequences=False,
                    input_shape=(None, in_neurons)))
    model.add(Dense(out_neurons, input_dim=hidden_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    # print(model.summary())


    # if os.path.exists("model.keras"):
        # model = load_model("model.keras")

    ## Train the LSTM model ##
    model.fit(true_X, true_Y, batch_size=2000, epochs=1, validation_split=0.05)
    model.save("model.keras")


    # predicted = model.predict(X)
    # print(predicted.shape)
else:
    for idx, transform in enumerate(transformations):
        start_end_indices = list(transformation_indices.values())
        seq_len = 1 + start_end_indices[idx][1] - start_end_indices[idx][0]
        X = transformed_hits_X[idx]
        Y = transformed_hits_Y[idx]
        in_neurons = seq_len
        out_neurons = seq_len
        hidden_neurons = 500


        # load the model if it exists
        if os.path.exists(transform + ".keras"):
            model = load_model(transform + ".keras")
        else:
            model = Sequential()
            model.add(LSTM(hidden_neurons, return_sequences=False,
                            input_shape=(None, in_neurons)))
            model.add(Dense(out_neurons, input_dim=hidden_neurons))
            model.add(Activation("linear"))
            model.compile(loss="mean_squared_error", optimizer="rmsprop")
            # print(model.summary())

        ## Train the LSTM model ##
        print("Training with the", transform, "tranformation")
        model.fit(X, Y, batch_size=2000, epochs=1, validation_split=0.05)
        model.save(transform + ".keras")


        # predicted = model.predict(X)
        # print(predicted.shape)


### viz ###
# import matplotlib.pylab as plt
# plt.scatter(predicted[:,0], predicted[:,1], color="red", s=1)
# plt.scatter(y[:,0], y[:,1], color="blue", s=1)
# plt.legend(["Predicted", "Actual"])
# plt.show()



#### old Save to / Load from File code ####
# if not os.path.exists("all_tracks.npy"):
#     for i in range(len(sorted_truth)):
#         truth = np.asarray(sorted_truth.iloc[i])
#         if truth[1] == 0:
#             continue
#         if truth[1] != current_pid and current_pid != -1:
#             current_pid = truth[1]
#             if len(track) == 0: continue
#             track = np.asarray(track)
#             # sort by distance from vertex
#             track = track[track[:,9].argsort()]
#             print("hit IDs?")
#             print(track[track[:,9].argsort()][:, 0])
#             true_tracks.append(track)
#             track = []
#         else:
#             current_pid = truth[1]
#             particle = np_particles[np_particles[:, 0] == current_pid]
#             particle = np.squeeze(particle)
            
#             # Add a "distance from vertex" column to each hit
#             #   enables sorting hits within a track
#             #   (i.e. 1st hit, 2nd hit, etc.) 
#             diff = np.subtract(particle[1:4], truth[2:5])
#             dist = np.linalg.norm(diff)
#             truth = np.append(truth, [dist])
            
#             track.append(truth)
#     np.save("all_tracks.npy", true_tracks)
# else:
#     true_tracks = np.load("all_tracks.npy")
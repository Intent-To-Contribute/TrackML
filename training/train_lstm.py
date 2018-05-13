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
transforms.dbscan_trans(hits)
transforms.spherical(hits)
transforms.cylindrical(hits)
transforms.normalize(hits)
transforms.standard(hits)
# print("hits", hits)

# create a mapping from columns names to index
column_index = {}
for column in hits.head():
    print(column)
    column_index[column] = hits.columns.get_loc(column)
print(column_index)

### group hits into their true tracks ###
sorted_truth = truth.sort_values("particle_id")
true_tracks = []
track = []
current_pid = -1
np_particles = np.asarray(particles)
np_hits = np.asarray(hits)


hit_tracks = []
for i in range(len(sorted_truth)):
    truth = np.asarray(sorted_truth.iloc[i])
    if truth[1] == 0:
        continue
    if truth[1] != current_pid and current_pid != -1:
        current_pid = truth[1]
        if len(track) == 0: continue
        track = np.asarray(track)
        # sort by distance from vertex
        track = track[track[:,9].argsort()]
        true_tracks.append(track)

        # get Hits from hit_id's,
        #   first column in truth file
        hitIDs = track[track[:,9].argsort()][:, 0]
        hit_track = []
        for hitID in hitIDs:
            hit_track.append(np_hits[int(hitID)])
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

print("finished initializing true_tracks")

print("number of true tracks", len(true_tracks))
print("shape of true tracks", np.asarray(true_tracks).shape)
print("number of hit tracks", len(hit_tracks))
print("shape of hit tracks", np.asarray(hit_tracks).shape)


### Format input sequences X and output Y ###
# --- making input vectors ---
X = []
Y = []
for track in true_tracks:
    for i in range(1, len(track)-1):
        x_hit = np.zeros((0, 3))
        for z in range(i):
            hit_to_add = np.asarray(track[z][2:5]).reshape(1,3)
            x_hit = np.concatenate((x_hit, hit_to_add))
        X.append(x_hit)
        Y.append(track[i+1][2:5])

print("finished making input vectors")

X = np.asarray(X)
Y = np.asarray(Y)

import tensorflow as tensorflow
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

# random inputs -- before padding
for i in range(10):
    rand_input = np.random.choice(X)
    # print("random input vector", rand_input)
    print("before padding -- input vector shape", rand_input.shape)

# pad sequences with 0's
X = keras.preprocessing.sequence.pad_sequences(X, maxlen=None, dtype=float)

# random inputs -- after padding
print("after padding")
rand_inputs = np.random.randint(X.shape[0], size=10)
for rand_input in rand_inputs:
    print("input vector")
    print(X[rand_input])
    print("output point")
    print(Y[rand_input])

print("batch input shape", X.shape)

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

model.fit(X, Y, batch_size=2000, epochs=1, validation_split=0.05)
model.save("model.keras")


predicted = model.predict(X)
print(predicted.shape)


### viz ###
# import matplotlib.pylab as plt
# plt.scatter(predicted[:,0], predicted[:,1], color="red", s=1)
# plt.scatter(y[:,0], y[:,1], color="blue", s=1)
# plt.legend(["Predicted", "Actual"])
# plt.show()



#### Save to / Load from File code ####
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
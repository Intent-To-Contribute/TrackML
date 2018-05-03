import tensorflow as tensorflow
import os, sys
import csv
import pandas as pd
import numpy as np
from trackml.dataset import load_event


### load data ###
hits, cells, particles, truth = load_event('../..//Data/train_100_events/event000001052')


### group hits into their true tracks ###
sorted_truth = truth.sort_values("particle_id")
true_tracks = []
track = []
current_pid = -1
if not os.path.exists("all_tracks.npy"):
    for i in range(len(sorted_truth)):
        line = sorted_truth.iloc[i]
        if line[1] == 0:
            continue
        if line[1] != current_pid:
            true_tracks.append(np.asarray(track))
            track = []
            current_pid = line[1]
        else:
            track.append(line)
    np.save("all_tracks.npy", true_tracks)
else:
    true_tracks = np.load("all_tracks.npy")

print("finished initializing true_tracks")


### Format input sequences X and output Y ###
max_len = 0
X = []
Y = []

# --- grabbing max(nhits) ---
for track in true_tracks:
    if len(track) > max_len:
        max_len = len(track)

# --- making input vectors ---
for track in true_tracks:
    curr_idx = 0
    for i in range(1, max_len-1):
        x_hit = np.zeros((max_len, 6))
        if i < len(track)-1:
            for z in range(i):
                x_hit[max_len-i+z] = track[z][2:8]
            X.append(x_hit)
            # Y.append(track[i+1][2:8])
            Y.append(track[i+1][2:5])
        

X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape, Y.shape)

print(X[0])
print(Y[0])


import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

### train model ###
seq_len = max_len
in_neurons = 6
out_neurons = 3
hidden_neurons = 500

model = Sequential()
# model.add(LSTM(hidden_neurons, return_sequences=False,
				# input_shape=(None, in_neurons)))
model.add(LSTM(hidden_neurons, return_sequences=False,
                input_shape=(seq_len, in_neurons)))
model.add(Dense(out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

print(model.summary())


model.fit(X, Y, batch_size=2000, epochs=1, validation_split=0.05)
model.save("model.keras")
# model = load_model("model.keras")


predicted = model.predict(X)
print(predicted.shape)


### viz ###
# import matplotlib.pylab as plt
# plt.scatter(predicted[:,0], predicted[:,1], color="red", s=1)
# plt.scatter(y[:,0], y[:,1], color="blue", s=1)
# plt.legend(["Predicted", "Actual"])
# plt.show()


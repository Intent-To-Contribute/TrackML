import tensorflow as tensorflow
import os, sys
import csv
import pandas as pd
import numpy as np
from trackml.dataset import load_event

hits, cells, particles, truth = load_event('../..//Data/train_100_events/event000001052')

sorted_truth = truth.sort_values("particle_id")
#print(sorted_truth)

true_tracks = []
track = []

current_pid = -1
if not os.path.exists("all_tracks.npy"):
    for i in range(len(sorted_truth)):
        line = sorted_truth.iloc[i]
        if line[1] != current_pid:
            true_tracks.append(np.asarray(track))
            track = []
            current_pid = line[1]
        else:
            track.append(line)
    np.save("all_tracks.npy", true_tracks)
else:
    true_tracks = np.load("all_tracks.npy")

print("finished initializeing true_tracks")

max_len = 0
X = []
Y = []
for track in true_tracks:
    if len(track) > max_len:
        max_len = len(track)
for track in true_tracks:
    curr_idx = 0
    for i in range(max_len-1):
        x_hit = np.zeros((max_len, 6))
        if i < len(track)-1:
            for z in range(i):
                x_hit[max_len-i+z] = track[z][2:8] 
            Y.append(track[i+1])
            X.append(x_hit)
        

X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape, Y.shape)

# print(true_tracks)

"""
x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

in_out_neurons = 2
hidden_neurons = 300

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
				input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

print(model.summary())


# model.fit(x, y, batch_size=20, epochs=10, validation_split=0.05)
# model.save("model.keras")
model = load_model("model.keras")


predicted = model.predict(x)
print(predicted.shape)

import matplotlib.pylab as plt
plt.scatter(predicted[:,0], predicted[:,1], color="red", s=1)
plt.scatter(y[:,0], y[:,1], color="blue", s=1)
plt.legend(["Predicted", "Actual"])
plt.show()
"""

import os, sys
import csv
import pandas as pd
import numpy as np
from trackml.dataset import load_event


### load data ###
hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')


### group hits into their true tracks ###
sorted_truth = truth.sort_values("particle_id")
true_tracks = []
track = []
current_pid = -1
np_particles = np.asarray(particles)
if not os.path.exists("all_tracks.npy"):
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
            track = []
        else:
            current_pid = truth[1]
            particle = np_particles[np_particles[:, 0] == current_pid]
            particle = np.squeeze(particle)
            diff = np.subtract(particle[1:4], truth[2:5])
            dist = np.linalg.norm(diff)
            truth = np.append(truth, [dist])
            track.append(truth)
    np.save("all_tracks.npy", true_tracks)
else:
    true_tracks = np.load("all_tracks.npy")

print("finished initializing true_tracks")


### Format input sequences X and output Y ###
max_len = 0
final_class_X = []
final_class_Y = []

# --- grabbing max(nhits) ---
for track in true_tracks:
    if len(track) > max_len:
        max_len = len(track)

print(max_len)

# --- making input vectors ---
for track in true_tracks:
    for i in range(1, max_len):
        x_hit = np.zeros((max_len, 3))

        if i < len(track):
            for z in range(i):
                x_hit[max_len-i+z] = track[z][2:5]
            final_class_X.append(x_hit)
            if i == len(track)-1:
                final_class_Y.append(np.asarray([1, 1]))
            else:
                final_class_Y.append(np.asarray([0, 1]))


final_class_X = np.asarray(final_class_X)
final_class_Y = np.asarray(final_class_Y)
print(final_class_X.shape, final_class_Y.shape)

print(final_class_X[0])
print()
print(final_class_Y[0])


### train model ###
import tensorflow as tensorflow
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

seq_len = max_len
in_neurons = 3
out_neurons = 2
hidden_neurons = 500

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
                input_shape=(seq_len, in_neurons)))
model.add(Dense(out_neurons, input_dim=hidden_neurons))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

print(model.summary())


# if os.path.exists("model.keras"):
    # model = load_model("model.keras")

model.fit(final_class_X, final_class_Y, batch_size=2000, epochs=1, validation_split=0.05)
model.save("final_classifier.keras")


predicted = model.predict(final_class_X)
print(predicted.shape)


### viz ###
# import matplotlib.pylab as plt
# plt.scatter(predicted[:,0], predicted[:,1], color="red", s=1)
# plt.scatter(y[:,0], y[:,1], color="blue", s=1)
# plt.legend(["Predicted", "Actual"])
# plt.show()


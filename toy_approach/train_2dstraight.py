import tensorflow as tensorflow
import os, sys
import csv
import pandas as pd
import numpy as np

# dataframe = pd.read_csv('gen_data/2d_straight.csv', engine='python')
dataframe = pd.read_csv('gen_data/2d_straight.full.csv', engine='python')

dataset = dataframe.values

hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')
# print (dataset[0:20,:].shape)
# print (dataset[0:20,:])
# print (dataset.reshape(10, 10, 4))
# print (dataset)
# print(len(dataset))

dataset = dataset.reshape(int(len(dataset)/10), 10, 4)
dataset = dataset[1500:1800]

data = []
x = []
y = []
for track in dataset:
	# zeroes = pd.DataFrame(0)
	zeros = np.zeros(shape=(9, 2))
	# print (zeros)
	vals = track[:,2:]
	vals = np.insert(vals, 0, zeros, axis=0)
	# print(vals.shape)	

	for i in range(9):
		x.append(vals[i:i+10])
		y.append(vals[i+10])
		# print(vals[i:i+10])
		# print(vals[i+10])

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

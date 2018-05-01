import keras
import tensorflow as tf
import os, sys

# print("python:{}, keras:{}, tensorflow:{}".format(sys.version, keras.__version__, tf.__version__))

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 2
hidden_neurons = 300

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
				input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

# model.summary()

import pandas as pd
from random import random

flow = (list(range(1, 10, 1)) + list(range(10, 1, -1))) * 1000
pdata = pd.DataFrame({"a":flow, "b":flow})
pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:] * random()  # some noise

import numpy as np

def _load_data(data, n_prev = 5):
	"""
	data should be pd.DataFrame()
	"""
	print (data.shape)
	docX, docY = [], []
	for i in range(len(data) - n_prev):
		docX.append(data.iloc[i:i+n_prev].as_matrix())
		docY.append(data.iloc[i+n_prev].as_matrix())
	alsX = np.array(docX)
	alsY = np.array(docY)

	print("alsX shape", alsX.shape)
	print("alsY shape", alsY.shape)
	print("alsX", alsX[0])
	print("alsY", alsY[0])
	return alsX, alsY

def train_test_split(df, test_size=0.1):
	"""
	This just splits data to training and testing parts
	"""
	ntrn = round(len(df) * (1 - test_size))

	X_train, y_train = _load_data(df.iloc[0:ntrn])
	X_test, y_test = _load_data(df.iloc[ntrn:])

	return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = train_test_split(data) # retrieve data

# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=100, epochs=1, validation_split=0.05)

predicted = model.predict(X_test)
print ("shape", predicted.shape)
print (predicted[0])
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (17, 9)
plt.plot(predicted[:100][:,0], "--")
plt.plot(predicted[:100][:,1], "--")
plt.plot(y_test[:100][:,0], ":")
plt.plot(y_test[:100][:,1], ":")
plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"])
plt.show()

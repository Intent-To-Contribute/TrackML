import os, sys
import csv
import pandas as pd
import numpy as np
from trackml.dataset import load_event, load_dataset

from data_formatter import DataFormatter
import transforms, dataset_path

from sklearn.preprocessing import StandardScaler
import tensorflow as tensorflow
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

### load data ###
#hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')
#hits, cells, particles, truth = load_event('../../Data/train_full/event000001052')

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

transformation_list = "\n" + "\n".join(str(idx+1)+". "+t for idx, t in enumerate(transformations))
transformation_idx = input(transformation_list + "\nChoose which transform to train: ")
transformation_idx = int(transformation_idx)-1
num_epochs = input("Input how many epochs you want to train: ")
num_epochs = int(num_epochs)
seq_len = input("Input the sequence length you want to train: ")
seq_len = int(seq_len)


formatter = DataFormatter()


## Create the LSTM model ##
# start_end_indices = list(transformation_indices.values())
# tuple_len = 1 + start_end_indices[transformation_idx][1] - start_end_indices[transformation_idx][0]
# X = transformed_hits_X[transformation_idx]
# Y = transformed_hits_Y[transformation_idx]
tuple_len = 3
in_neurons = tuple_len
out_neurons = tuple_len
hidden_neurons = 500

# load the model if it exists
if os.path.exists(transformations[transformation_idx] + ".keras"):
    model = load_model(transformations[transformation_idx] + ".keras")
else:
    model = Sequential()
    model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=(None, in_neurons)))
    model.add(Dense(out_neurons, input_dim=hidden_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    # print(model.summary())



# Train on events
for event_id, hits, cells, particles, truth in load_dataset(dataset_path.get_path()):
    print("-"*15, "processing event #", event_id, "-"*15)


    transform = transformations[transformation_idx]
    getattr(transforms, transform)(hits, True) # equivalent to transforms.transform(hits, True)
    
    
    true_tracks, hit_tracks, max_len = formatter.getSortedTracks(particles, truth, hits)
    x, y = formatter.getInputOutput(hit_tracks, 1, 3, seq_len, seq_len, 18)
    

    ## Train the LSTM model ##
    print("Training with the", transformations[transformation_idx], "tranformation")
    model.fit(x, y, batch_size=2000, epochs=num_epochs, validation_split=0)
    model.save(transformations[transformation_idx] + str(seq_len) +  ".keras")


## debugging code ##

# sanity checks for shape
# printShapes = False
# if (printShapes):
#     print()
#     print("-"*20 + " random input vector shapes before padding " + "-"*20)
#     for i in range(3):
#         rand_input = np.random.choice(true_X)
#         print("true input", rand_input.shape)

#     for transformed_X in transformed_hits_X:
#         for i in range(3):
#             rand_input = np.random.choice(transformed_X)
#             print("transformed input", rand_input.shape)

# sanity checks for shape
# if (printShapes):
#     print()
#     print("-"*20 + " random input vector shapes after padding " + "-"*20)
#     rand_inputs = np.random.randint(true_X.shape[0], size=3)
#     for rand_input in rand_inputs:
#         print("true input", true_X[rand_input].shape)
#         print("true output", true_Y[rand_input].shape)

#     for rand_input in rand_inputs:
#         for transformed_X in transformed_hits_X:
#             print("transformed input", transformed_X[rand_input].shape)
#         for transformed_Y in transformed_hits_Y:
#             print("transformed output", transformed_Y[rand_input].shape)
            
#     print("batch input shape -- true", true_X.shape)
#     for transformed_X in transformed_hits_X:
#         print("batch input shape -- transformed", transformed_X.shape)
#     print()

# # create a mapping from column_names to the corresponding index (in 'hits')
# column_index = {}
# for column_name in hits.head():
#     column_index[column_name] = hits.columns.get_loc(column_name)

# ## mapping from transformation to start, end indices ##
# transformation_indices = {
#     "dbscan_trans"  : [column_index['1_db'], column_index['3_db']],
#     "spherical"     : [column_index['1_sph'], column_index['3_sph']],
#     "cylindrical"   : [column_index['1_cyl'], column_index['3_cyl']],
#     "normalize"     : [column_index['1_norm'], column_index['4_norm']],
#     "standard"      : [column_index['1_ss'], column_index['3_ss']],
#     "identity"      : [column_index['1_id'], column_index['3_id']]
# }

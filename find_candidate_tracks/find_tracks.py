import sys
import numpy as np
import matplotlib.pyplot as plt
from trackml.dataset import load_event

### load data ###
hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')
true_tracks = np.load("../port_toy/all_tracks.npy")

### track lengths ###
# lengths = np.zeros(20)
# for track in true_tracks[500:520]:
# for track in true_tracks:
	# print("track is length: " + str(len(track)))
	# lengths[len(track)] += 1
	# for hit in track:
		# print(hit[2:4])
	# print("\n")

# for (idx, num) in enumerate(lengths):
	# print(idx, num)
# plt.plot(range(20), lengths)
# plt.show()

### norms ###
# seeds = []
# norms = []
# mins = []
# for track in true_tracks:
# 	if (len(track) > 7):
# 		min = (sys.maxsize, None)
# 		for hit in track:
# 			norm = np.linalg.norm(hit[2:4])
# 			# print(norm)
# 			norms.append(norm)
# 			if (min[0] > norm): min = (norm, hit)
# 		print(min[0])
# 		mins.append(min[0])

# plt.hist(mins, bins=20)
# plt.show()
# plt.hist(norms, bins=20)
# plt.show()

### hits ###
# hits = []
# xMax = -sys.maxsize
# yMax = -sys.maxsize
# zMax = -sys.maxsize
# xMin = sys.maxsize
# yMin = sys.maxsize
# zMin = sys.maxsize
# for track in true_tracks:
# 	for hit in track:
# 		if (xMax < hit[2]): xMax = hit[2]
# 		if (yMax < hit[3]): yMax = hit[3]
# 		if (zMax < hit[4]): zMax = hit[4]
# 		if (xMin > hit[2]): xMin = hit[2]
# 		if (yMin > hit[3]): yMin = hit[3]
# 		if (zMin > hit[4]): zMin = hit[4]
# 		hits.append(hit)


### seeds ###
import csv

seed_file = open("SeedCandidates.txt", "r")
seed_hits = []
hits = np.asarray(hits)
for seed_id in seed_file:
	seed_id = int(float(seed_id.strip()))
	print(seed_id)
	seed_hit = hits[hits[:,0] == seed_id][0]
	seed_hits.append(seed_hit[1:7])

print(len(seed_hits))

x = []
for seed_hit in seed_hits:
	input_vector = np.zeros((18, 6))
	print(seed_hit)
	# input_vector[17] = seed_hit
	# x.append(input_vector)

x = np.asarray(x)


### use the model ###
import tensorflow as tensorflow
import keras
from keras.models import load_model

model = load_model("attempt1.keras")
print(model.summary())

y = model.predict(x)
print(y)

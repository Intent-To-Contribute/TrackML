import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from trackml.dataset import load_event
import pdb

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
# 	if (len(track) > 71):
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
#hits = []
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
#		hits.append(hit)

### seeds ###
import csv

seed_file = open("SeedCandidates.txt", "r")
seed_hits = []
hits = np.asarray(hits)
for seed_id in seed_file:
	seed_id = int(float(seed_id.strip()))
	seed_hit = hits[hits[:,0] == seed_id][0]
	seed_hits.append(seed_hit[1:4])

print(len(seed_hits))

x = []
for seed_hit in seed_hits:
	input_vector = np.zeros((18, 3))
	input_vector[17] = seed_hit
	x.append(input_vector)

x = np.asarray(x)
if not os.path.exists("guesses.npy"):
    import tensorflow as tensorflow
    import keras
    from keras.models import load_model

   ## use the model ###
    model = load_model("3in_3out.keras")
    print(model.summary())

    y = model.predict(x)
    print(y)
    np.save("guesses.npy", y)
else:
    y = np.load("guesses.npy")

def find_nearest_hit(hits, guess):
    min_dist = sys.maxsize
    closest_hit = hits[0]
    for hit in hits:
        dist = np.linalg.norm(np.subtract(hit[1:4], guess))
        if dist < min_dist:
            min_dist = dist
            closest_hit = hit
    return closest_hit


# create voxelized hits
xRange = xMax - xMin
yRange = yMax - yMin
zRange = zMax - zMin
n = 100
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

print(voxels)

#hit = hits[5]
#xHit = hit[1]
#yHit = hit[2]
#zHit = hit[3]
#i = int(n * ((xHit - xMin) / xRange))
#j = int(n * ((yHit - yMin) / yRange))
#k = int(n * ((zHit - zMin) / zRange))
#
#print(voxels[i][j][k])
#print(len(voxels[i][j][k]))

next_hits = []
counter = 0
for guess in y:    
    xHit = guess[0]
    yHit = guess[1]
    zHit = guess[2]
    i = int(n * ((xHit - xMin) / xRange))
    j = int(n * ((yHit - yMin) / yRange))
    k = int(n * ((zHit - zMin) / zRange))

    possible_nearest_hits = []
    for ii in range(min(0,i-1), max(i+2,n+1)):
        for jj in range(min(0,j-1), max(j+2,n+1)):
            for kk in range(min(0,k-1), max(k+2,n+1)):
                print(voxels[ii][jj][kk])
                if voxels[ii][jj][kk] != 0:
                    possible_nearest_hits.append(voxels[ii][jj][kk])
                    pdb.set_trace()

    print(possible_nearest_hits.shape)
    next_hit = find_nearest_hit(possible_nearest_hits, guess)
    next_hits.append(next_hit)
    print(str(counter) + "/" + str(len(y)))
    counter += 1

print(next_hits)



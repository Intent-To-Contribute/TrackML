import sys
import numpy as np
import matplotlib.pyplot as plt

true_tracks = np.load("../port_toy/all_tracks.npy")
print("hi")
print(true_tracks[1:10])
print("bye")


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

seeds = []
norms = []
mins = []
for track in true_tracks:
	if (len(track) > 7):
		min = (sys.maxsize, None)
		for hit in track:
			norm = np.linalg.norm(hit[2:4])
			# print(norm)
			norms.append(norm)
			if (min[0] > norm): min = (norm, hit)
		print(min[0])
		mins.append(min[0])

plt.hist(mins, bins=20)
plt.show()
# plt.hist(norms, bins=20)
# plt.show()

# import tensorflow as tensorflow
# import keras
# from keras.models import load_model

# model = load_model("attempt1.keras")
# print(model.summary())


from octree import Octree

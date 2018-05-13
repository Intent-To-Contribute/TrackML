import os, sys
import csv
import pandas as pd
import numpy as np
from trackml.dataset import load_event

### load data ###
hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')

### group hits into their true tracks ###
sorted_truth = truth.sort_values("particle_id")
sorted_tracks = []
track = []
current_pid = -1
np_particles = np.asarray(particles)
if not os.path.exists("sorted_tracks_1052.npy"):
    for i in range(len(sorted_truth)):
        truth = np.asarray(sorted_truth.iloc[i])

        # skip noise points
        if truth[1] == 0:
            continue

        if truth[1] != current_pid and current_pid != -1:
            current_pid = truth[1]
            if len(track) == 0: continue
            track = np.asarray(track)
            # sort by distance from vertex
            track = track[track[:,9].argsort()]
            sorted_tracks.append(track)
            track = []
        else:
            current_pid = truth[1]
            particle = np_particles[np_particles[:, 0] == current_pid]
            particle = np.squeeze(particle)
            diff = np.subtract(particle[1:4], truth[2:5])
            dist = np.linalg.norm(diff)
            truth = np.append(truth, [dist])
            track.append(truth)
    np.save("sorted_tracks_1052.npy", sorted_tracks)
else:
    sorted_tracks = np.load("sorted_tracks_1052.npy")


first_hit_norms = []
non_first_or_second_hit_norms = []
second_hit_norms = []
for track in sorted_tracks:
    first_hit_norms.append(np.linalg.norm(track[0][2:4]))    
    if (len(track) > 1): second_hit_norms.append(np.linalg.norm(track[1][2:4]))
    for truth_hit in track[2:]:
        non_first_or_second_hit_norms.append(np.linalg.norm(truth_hit[2:4]))
    # for truth_hit in track[1:]:
        # non_first_hit_norms.append(np.linalg.norm(truth_hit[2:4]))

from matplotlib import pyplot
import numpy as np

bins = np.linspace(25, 200, 50)


# pyplot.hist([first_hit_norms, non_first_hit_norms], bins, label=['first hits', 'non-first hits'])
# pyplot.yscale('log', nonposy='clip')
pyplot.hist([first_hit_norms, second_hit_norms, non_first_or_second_hit_norms], bins, stacked=True, label=['first hits', 'second hits', 'non first or second hits'])
pyplot.legend(loc='upper right')
pyplot.title("xy-norms")
pyplot.show()

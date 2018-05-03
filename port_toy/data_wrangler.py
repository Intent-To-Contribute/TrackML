import sys
from trackml.dataset import load_event
import numpy as np
import pdb
from tempfile import TemporaryFile


# -- reading in data --
hits, cells, particles, truth = load_event('../..//Data/train_100_events/event000001052')



sorted_truth = truth.sort_values("particle_id")
#print(sorted_truth)

true_tracks = []
track = []

current_pid = -1
for i in range(len(sorted_truth)):
    line = sorted_truth.iloc[i]
    if line[1] != current_pid:
        true_tracks.append(track)
        track = []
        current_pid = line[1]
    else:
        track.append(line)

#true_tracks_array = np.asarray(true_tracks)
#print(len(true_tracks))
#print(len(np.unique(np.asarray(truth["particle_id"]))))
#
#outfile = open("true_tracks_1052.txt","w+")
#np.save(outfile, true_tracks_array)

print(true_tracks[5])

#for track in true_tracks:
#    print("hi")

#for i in range(len(truth)):
#  # print(hits.iloc[i])
#  
#  for val in truth.iloc[i]:
#    print(val),
#    print("\n")
#    if val





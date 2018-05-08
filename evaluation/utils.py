import numpy as np
import sys


def find_nearest_hit(hits, guess):
    # if there are nearby hits
    min_dist = sys.maxsize
    closest_hit = hits[0]
    for hit in hits:
        dist = np.linalg.norm(np.subtract(hit[1:4], guess))
        if dist < min_dist:
            min_dist = dist
            closest_hit = hit
    return closest_hit


def scan_voxels_for_hits(voxels, n, i, j, k):
    window = 1
    high = window + 1
    possible_nearest_hits = []
    while len(possible_nearest_hits) == 0:
        high = window + 1
        for ii in range(max(0,i-window), min(i+high,n+1)):
            for jj in range(max(0,j-window), min(j+high,n+1)):
                for kk in range(max(0,k-window), min(k+high,n+1)):
                    if voxels[ii][jj][kk] != 0:
                        for hit in voxels[ii][jj][kk]:
                            possible_nearest_hits.append(hit)
        window += 1
    return np.asarray(possible_nearest_hits)


def stop_or_no(model, input_vector, classification):
    # call the model
    pred = model.predict(input_vector)
    return np.argmax(pred)


def sort_tracks(hits, truth, particles):
    sorted_truth = truth.sort_values("particle_id")
    sorted_tracks = []
    track = []
    current_pid = -1
    np_particles = np.asarray(particles)
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

    return sorted_tracks
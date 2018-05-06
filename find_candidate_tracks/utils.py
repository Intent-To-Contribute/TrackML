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

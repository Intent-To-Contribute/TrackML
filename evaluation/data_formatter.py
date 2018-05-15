import sys, math
import numpy as np
from trackml.dataset import load_event

class DataFormatter:
    def __init__(self):
        pass

    def getSortedTracks(self, particles, truth, hits, eventID=None):
        true_tracks = []
        hit_tracks = []
        sorted_truth = truth.sort_values("particle_id")
        np_hits = np.asarray(hits)
        np_particles = np.asarray(particles)
        current_pid = -1
        track = []

        total = len(truth.groupby('particle_id'))
        percent = int(total / 100)
        print("Sort tracks... ", end="")

        max_len = -1
        for (i, (particle_id, true_track)) in enumerate(truth.groupby('particle_id')):
            if i % percent == 0: print("\rSort tracks... " + str(int(100*i / total)) + "%", end="")
            if particle_id == 0: continue
            if len(true_track) < 4: continue
            if len(true_track) > max_len: max_len = len(true_track)
            vertex = np_particles[np_particles[:,0] == particle_id][0][1:4]

            # vertex = particle[1:4]
            np_true_track = true_track.values
            t_points = np_true_track[:,2:5]
            dists = np.transpose([np.linalg.norm(vertex-t_points, axis=1)])
            true_track = np.append(np_true_track, dists, axis=1)
            true_track = true_track[true_track[:,9].argsort()]
            true_tracks.append(true_track)
            hit_track = [np_hits[np_hits[:,0] == true_hit[0]] for true_hit in true_track]
            hit_tracks.append(np.asarray(hit_track))

        print("\rSort tracks... 100%", end="")
        print("\n", end="")
        return true_tracks, hit_tracks, max_len-1

    def getInputOutput(self, tracks, minLength=-sys.maxsize, maxLength=sys.maxsize):
        x = []
        y = []
        for track in tracks:
            for i in range(max(1, minLength), min(len(track)-1, maxLength+1)):
                x_hit = np.zeros((0, 3))
                for z in range(i):
                    hit_to_add = np.asarray(track[z][2:5]).reshape(1,3)
                    x_hit = np.concatenate((x_hit, hit_to_add))
                x.append(x_hit)
                y.append(track[i+1][2:5])
        
        if (len(x) == 0): return np.empty(0), np.empty(0)
        from keras.preprocessing.sequence import pad_sequences
        x = pad_sequences(x, maxlen=None, dtype=np.float64)
        y = np.asarray(y)

        return x, y


# test
if __name__ == "__main__":
    path_to_dataset = "../../Data/train_100_events/"
    event_path = "event000001052"
    hits, cells, particles, truth = load_event(path_to_dataset + event_path)

    formatter = DataFormatter()
    true_tracks, hits_tracks = formatter.getSortedTracks(particles, truth, hits)
    print("sorted tracks shape", len(true_tracks), len(hits_tracks))

    true_x, true_y = formatter.getInputOutput(true_tracks, 3)
    hits_x, hits_y = formatter.getInputOutput(hits_tracks, 3)
    print("input shape", true_x.shape, hits_x.shape)
    print("output shape", true_y.shape, hits_y.shape)

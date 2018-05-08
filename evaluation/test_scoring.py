# To ogenerate a random test submission from truth information and compute the expected score:

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')

shuffled = shuffle_hits(truth, 0.70) # n% probability to reassign a hit
score = score_event(truth, shuffled)
#print(shuffled)
#print(shuffled.head())
#print(shuffled.dtypes)
print(score)


#import pandas as pd
#import numpy as np
#
#
## stole this from the dbscan submission ~~~
#def create_one_event_submission(event_id, hits, labels):
#    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
#    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
#    return submission
#
#submission = np.load("submission_1052.npy")
#
#print(submission.shape)
#submission = np.unique(submission, axis=0)
#print(submission.shape)
#
#temp = []
#ids = {}
#for track in submission:
#    seent = False
#    for hit in track:
#        if hit in ids.keys():
#           seent = True 
#        else:
#            ids[hit] = 1
#    if not seent:
#        temp.append(track)
#
#print(len(temp))
#submission = np.asarray(temp)
#
##submission = np.sort(submission, axis=0)
#submission = pd.DataFrame(np.asarray(submission), columns = ["hit_id", "track_id"])
##print(submission.dtypes)
#submission.sort_values("hit_id", inplace=True)
##print(submission.head())
##print(submission)
#
#score = score_event(truth, submission)
#print(score)

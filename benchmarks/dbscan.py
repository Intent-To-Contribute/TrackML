import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

path_to_train = "../../data/train_100_events"

event_prefix = "event000001000"

hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

hits.head()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering

class Clusterer(object):
    def __init__(self, eps):
        self.eps = eps        

    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        # "normalize" transformation
        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r
        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r
        hits['r'] = r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        # X = ss.fit_transform(hits[['x2', 'y2', 'z2', 'r']].values)
        
        return X
    
    def predict(self, hits):
        
        X = self._preprocess(hits)
        
        cl = DBSCAN(eps=self.eps, min_samples=1, algorithm='kd_tree')
        labels = cl.fit_predict(X)
        
        return labels

model = Clusterer(eps=0.008)
labels = model.predict(hits)
print(labels)


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

#####
if (False):
# if (True):
    num_range = 6
    base = 0.0070317
    step_size = 0.005*base / num_range
    max_score = 0
    eps_with_max_score = 0
    for i in range(1, num_range+1):
        eps = base + ((i-0.5*num_range)*step_size)
        eps = abs(eps)
        print("eps:\t", eps)
        model = Clusterer(eps=eps)
        labels = model.predict(hits)

        submission = create_one_event_submission(0, hits, labels)
        score = score_event(truth, submission)
        print("score:\t", score, "\n")

        if (score > max_score):
            max_score = score
            eps_with_max_score = eps

    model = Clusterer(eps=eps_with_max_score)
    labels = model.predict(hits)

    print("Best eps was", eps_with_max_score, "with a score of", max_score)
#####

submission = create_one_event_submission(0, hits, labels)
score = score_event(truth, submission)

print("Your score: ", score)


## Recognize tracks in all events of a dataset
# load_dataset(path_to_train, skip=1000, nevents=5)
dataset_submissions = []
dataset_scores = []

for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
    # Track pattern recognition
    # model = Clusterer(eps=0.008)
    model = Clusterer(eps=0.0070317)
    labels = model.predict(hits)
        
    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)
    dataset_submissions.append(one_submission)
    
    # Score for the event
    score = score_event(truth, one_submission)
    dataset_scores.append(score)
    
    print("Score for event %d: %.3f" % (event_id, score))
    
print('Mean score: %.3f' % (np.mean(dataset_scores)))



## Create a submission
path_to_test = "../../test"
test_dataset_submissions = []

# create_submission = False # True for submission 
create_submission = True # True for submission 

if create_submission:
    for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

        # Track pattern recognition
        # model = Clusterer(eps=0.008)
        model = Clusterer(eps=0.0070317)
        labels = model.predict(hits)

        # Prepare submission for an event
        one_submission = create_one_event_submission(event_id, hits, labels)
        test_dataset_submissions.append(one_submission)
        
        print('Event ID: ', event_id)

    # Create submission file
    submussion = pd.concat(test_dataset_submissions, axis=0)
    submussion.to_csv('submission.csv.gz', index=False, compression='gzip')


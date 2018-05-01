from trackml.dataset import load_event
import pdb
#hits, cells, particles, truth = load_event('path/to/event000000123')
hits, cells, particles, truth = load_event('../..//Data/train_100_events/event000001052')
#print(hits)
#print(truth["particle_id"])
pdb.set_trace()
print(len(truth["particle_id"].unique))


# print(hits.iloc[0])

"""
for i in range(len(hits)):
  # print(hits.iloc[i])
  
  for val in hits.iloc[i]:
    print(val),

  print("\n")
"""

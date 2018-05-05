from trackml.dataset import load_event
import pdb
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-- data -=-=-=-=-=-=-=-=-=-=-=-=-=-
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
hits, cells, particles, truth = load_event('../../Data/train_100_events/event000001052')

print(particles.iloc[0])
print(len(particles))

vx = []
vy = []
vz = []
px = []
py = []
pz = []
e = []
nhits = []

p = []
r = []

for i in range(len(particles)):
    vx.append(particles.iloc[i][1])
    vy.append(particles.iloc[i][2])
    vz.append(particles.iloc[i][3])
    px.append(particles.iloc[i][4])
    py.append(particles.iloc[i][5])
    pz.append(particles.iloc[i][6])
    e.append(particles.iloc[i][7])
    nhits.append(particles.iloc[i][8])

    r_xy = ( particles.iloc[i][1]**2 + particles.iloc[i][2]**2 )**(.5)
    p_total = ( particles.iloc[i][4]**2 + particles.iloc[i][5]**2 + particles.iloc[i][6]**2 )**(.5)

    p.append(p_total)
    r.append(r_xy)



# ~~~~~~~~~ diagnostics  ~~~~~~~~~


# 1d histograms
n_bins = 1000
n, bins, patches = plt.hist(vy, n_bins, facecolor='g', alpha=0.5)
plt.show()


# 2d histograms

#n_bins = 10
#plt.hist2d(r, p, bins=n_bins, norm=LogNorm())
#plt.colorbar()
#plt.show()



# <-- eof

import sys, math
import numpy as np
from trackml.dataset import load_event

class Voxels:
    def __init__(self, points, numBins=100):
        self.numBins = numBins
        self.numPoints = len(points)

        self.minX = points.x.min()
        self.minY = points.y.min()
        self.minZ = points.z.min()
        self.maxX = points.x.max()
        self.maxY = points.y.max()
        self.maxZ = points.z.max()

        self.xRange = self.maxX - self.minX
        self.yRange = self.maxY - self.minY
        self.zRange = self.maxZ - self.minZ

        self.bins = [[[[] for i in range(numBins)] for j in range(numBins)] for k in range(numBins)]

        # i = np.floor((numBins-1) * (points.x.values - self.minX) / self.xRange).astype(int)
        # j = np.floor((numBins-1) * (points.y.values - self.minY) / self.yRange).astype(int)
        # k = np.floor((numBins-1) * (points.z.values - self.minZ) / self.zRange).astype(int)

        i, j, k = self.getBinIndices(points.x.values, points.y.values, points.z.values)

        raw = points.values
        for idx in range(self.numPoints):
            self.bins[i[idx]][j[idx]][k[idx]].append(raw[idx])

    def getBinIndices(self, x, y, z):
        i = np.floor((self.numBins-1) * (x - self.minX) / self.xRange).astype(int)
        j = np.floor((self.numBins-1) * (y - self.minY) / self.yRange).astype(int)
        k = np.floor((self.numBins-1) * (z - self.minZ) / self.zRange).astype(int)

        return (i, j, k)

    def findClosestPoint(self, x, y, z):
        return None


    def getMaxBinCount(self):
        for i in range(self.numBins):
            for j in range(self.numBins):
                for k in range(self.numBins):
                    if len(self.bins[i][j][k]) > self.maxBinCount:
                        self.maxBinCount = len(self.bins[i][j][k])
        return maxBinCount

### test voxels
if __name__ == "__main__":
    path_to_dataset = "../../Data/train_100_events/"
    event_path = "event000001052"

    hits, cells, particles, truth = load_event(path_to_dataset + event_path)

    hit_voxels = Voxels(hits, 300)

    print(hit_voxels.getMaxBinCount())

import sys, math
import numpy as np
from trackml.dataset import load_event

class Voxels:
    def __init__(self, points, numBins=100):
        self.numBins = numBins
        self.numPoints = len(points)
        self.xIndex = points.columns.get_loc('x')
        self.yIndex = points.columns.get_loc('y')
        self.zIndex = points.columns.get_loc('z')

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
        self.usedIndices = set()
        
        i, j, k = self.getBinIndices(points.x.values, points.y.values, points.z.values)
        raw = points.values
        for idx in range(self.numPoints):
            self.usedIndices.add((i[idx], j[idx], k[idx]))
            # self.bins[i[idx]][j[idx]][k[idx]].append(raw[idx])
            self.bins[i[idx]][j[idx]][k[idx]].append(points.iloc[idx])

    def getBinIndices(self, x, y, z):
        i = np.floor((self.numBins-1) * (x - self.minX) / self.xRange).astype(int)
        j = np.floor((self.numBins-1) * (y - self.minY) / self.yRange).astype(int)
        k = np.floor((self.numBins-1) * (z - self.minZ) / self.zRange).astype(int)

        return i, j, k

    def findClosestPoint(self, x, y, z):
        i, j, k = self.getBinIndices(x, y, z)
        window = 1
        high = window + 1
        neighboring_points = []
        while len(neighboring_points) == 0:
            for ii in range(max(0,i-window), min(i+high,self.numBins+1)):
                for jj in range(max(0,j-window), min(j+high,self.numBins+1)):
                    for kk in range(max(0,k-window), min(k+high,self.numBins+1)):
                        if len(self.bins[ii][jj][kk]) != 0:
                            for point in self.bins[ii][jj][kk]:
                                neighboring_points.append(point)
            window += 1
            high = window + 1

        min_dist = sys.maxsize
        closest_point = neighboring_points[0]
        for point in neighboring_points:
            dist = math.sqrt((x - point.x)**2 + (y - point.y)**2 + (z - point.z)**2)
            if dist < min_dist:
                min_dist = dist
                closest_point = point

        return closest_point


    def getMaxBinCount(self):
        maxBinCount = 0
        for idx in self.usedIndices:
            if len(self.bins[idx[0]][idx[1]][idx[2]]) > maxBinCount:
                maxBinCount = len(self.bins[idx[0]][idx[1]][idx[2]])
        return maxBinCount

### test voxels
if __name__ == "__main__":
    path_to_dataset = "../../Data/train_100_events/"
    event_path = "event000001052"

    hits, cells, particles, truth = load_event(path_to_dataset + event_path)

    hit_voxels = Voxels(hits, 300)

    print(hit_voxels.getMaxBinCount())

    for i in range(100):
        print("closest point to ", hits.x.values[i], hits.y.values[i], hits.z.values[i])
        print(hit_voxels.findClosestPoint(hits.x.values[i], hits.y.values[i], hits.z.values[i]))

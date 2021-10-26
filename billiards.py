from matplotlib import lines
import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
import utils
import time

class PolygonOuterBillards:
    def __init__(self, nSides=7, radius=1, cutLength=25, includePolygon=True):
        self.nSides = nSides
        self.radius = radius
        self.cutLength = cutLength
        
        self.verts = utils.polygonVertices(nSides, radius=radius)
        self.cuts = self.getCuts(self.verts, cutLength, includePolygon)
        self.endPts = self.cuts[:,1,:]
        
    @staticmethod
    def getCuts(verts, cutLength, includePolygon=True):
        sideLength = linalg.norm(verts[1] - verts[0])
        cuts = []

        for i in range(len(verts)):
            dir = (verts[i] - verts[i-1]) / sideLength
            
            if includePolygon:
                cuts.append([
                    verts[i-1],
                    verts[i] + dir * cutLength
                ])
            else:
                cuts.append([
                    verts[i],
                    verts[i] + dir * cutLength
                ])
        
        return np.array(cuts, dtype=np.float64)
    
    def pointInRegion(self, points, region, includeEdges=True):
        """
        regions are 0 to nSides-1
        Points in the ith region should get reflected accross the ith vertex.
        """
        i, j = region, (region + 1) % self.nSides

        if includeEdges:
            return ((utils.sameSide(self.cuts[i], points, self.verts[j]) >= 0)
                  & (utils.sameSide(self.cuts[j], points, self.endPts[i]) >= 0))
        else:
            return ((utils.sameSide(self.cuts[i], points, self.verts[j]) > 0)
                  & (utils.sameSide(self.cuts[j], points, self.endPts[i]) > 0))


class BilliardPoints(PolygonOuterBillards):
    def generatePoints(self, count):
        """
        Generate an even distribution of points along the cuts
        """
        s = np.linspace(0, 1, count).reshape((-1, 1, 1))
        points = self.cuts[:,0,:] + s * (self.cuts[:,1,:] - self.cuts[:,0,:])
        return points.reshape((-1, 2))
    
    def transformPts(self, points, includeEdges=True):
        """
        Apply billiard transformation to points
        If includeEdges is False, points on cuts are discarded
        """
        M = np.array([[-1, 0],
                      [0, -1]])
        
        newPointSets = []
        for i in range(self.nSides):
            mask = self.pointInRegion(points, i, includeEdges)
            
            #Must transpose pts to multiply by matrix (and then transpose back)
            rotated = (M @ (points[mask,...] - self.verts[i]).T).T + self.verts[i]
            newPointSets.append(rotated)

        return np.concatenate(newPointSets)
    
    def iterator(self, seedPoints=100, iterations=100,
                 addPoints=True, removeDuplicates=True,
                 addPointsEvery=1, removeDuplicatesEvery=1):
        """
        Iterate the outer billiards map.
        Initially, each cut has seedPoints number of points.
          This can also be a set of points to use.
        Each iteration:
          seedPoints are added if addPoints=True (except first iteration)
          points are transformed
          duplicates are removed if removeDuplicates=True
        """
        if np.isscalar(seedPoints):
            seedPoints = self.generatePoints(int(seedPoints))
        
        points = seedPoints
        i = 0

        while i < iterations:
            if addPoints and i > 0 and i % addPointsEvery == 0:
                points = np.concatenate((points, seedPoints))
            
            points = self.transformPts(points)

            if removeDuplicates and i % removeDuplicatesEvery == 0:
                points = utils.removeDuplicatePts(points)
            
            yield points

            i += 1


class BilliardLines(PolygonOuterBillards):
    def transformLines(self, lines, includeEdges=True):
        """
        Apply billiard transformation to lines
        If includeEdges is False, lines on cuts are discarded
        Note: lines cannot cross a cut (please split then first)
        """
        M = np.array([[-1, 0],
                      [0, -1]])
        
        mid = utils.midpoint(lines)
        newLineSets = []
        for i in range(self.nSides):
            mask = self.pointInRegion(mid, i, includeEdges)
            
            rotated = utils.applyMatrixToLines(lines[mask,...] - self.verts[i], M) + self.verts[i]
            newLineSets.append(rotated)

        return np.concatenate(newLineSets)
    
    def splitLines(self, lines):
        """
        Split a set of line segments along the singularities.
        Afterward all lines lie in only one region
        """
        return utils.cutLines(lines, self.cuts, extend=(0, inf))
    
    def iterator(self, iterations=100, seedLines=None,
                 addPoints=True, removeDuplicates=True,
                 addPointsEvery=1, removeDuplicatesEvery=1):
        """
        Iterate the outer billiards map.
        Initially, the lines are the cut lines.
          This can also be a set of lines to use (if seedLines is set).
        Each iteration:
          seedLines are added if addLines=True (except first iteration)
          lines are cut
          lines are transformed
          overlaping lines are removed if removeOverlap=True
        """
        if seedLines is None:
            seedLines = self.cuts

        lines = seedLines
        i = 0

        while i < iterations:
            if addPoints and i > 0 and i % addPointsEvery == 0:
                lines = np.concatenate((lines, seedLines))
            
            lines = self.splitLines(lines)
            lines = self.transformLines(lines)

            if removeDuplicates and i % removeDuplicatesEvery == 0:
                lines = utils.mergeOverlapingSegments(lines)
            
            yield lines

            i += 1

if __name__ == "__main__":
    billiard = BilliardLines()

    fig = plt.figure()

    for i, points in enumerate(billiard.iterator(addPoints=False)):
        print(i, len(points))
        fig.clear()
        utils.plotPoints(points)
        plt.pause(0.1)
    
    fig.clear()
    utils.plotPoints(points)
    plt.show()
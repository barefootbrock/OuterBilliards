from matplotlib import lines
import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
import utils
import time

class BilliardPoints:
    def __init__(self, nSides=7, radius=1, cutLength=25, includePolygon=True):
        self.nSides = nSides
        self.radius = radius
        self.cutLength = cutLength
        
        self.verts = utils.polygonVertices(nSides, radius=radius)

        self.sideLength = linalg.norm(self.verts[1] - self.verts[0])

        endPts = []
        for i in range(nSides):
            dir = (self.verts[i] - self.verts[i-1]) / self.sideLength
            endPts.append(self.verts[i] + dir * cutLength)
        self.endPts = np.array(endPts)

        if includePolygon:
            self.cuts = np.array([[self.verts[i-1], endPts[i]] for i in range(nSides)]) #Include central polygon
        else:
            self.cuts = np.array([[self.verts[i], endPts[i]] for i in range(nSides)]) #do not include it
    
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
            j = (i + 1) % self.nSides
            k = (i + 2) % self.nSides

            if includeEdges:
                mask = ((utils.sameSide(self.cuts[i], points, self.verts[j]) >= 0)
                      & (utils.sameSide(self.cuts[j], points, self.endPts[i]) >= 0)
                      & ((utils.sameSide(self.cuts[j], points, self.endPts[i]) > 0)
                       | (utils.sameSide(self.cuts[k], points, self.verts[i]) > 0)))
            else:
                mask = ((utils.sameSide(self.cuts[i], points, self.verts[j]) > 0)
                      & (utils.sameSide(self.cuts[j], points, self.endPts[i]) > 0))
            
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

        yield points

        while i < iterations:
            if addPoints and i > 0 and i % addPointsEvery == 0:
                points = np.concatenate((points, seedPoints))
            
            points = self.transformPts(points)

            if removeDuplicates and i % removeDuplicatesEvery == 0:
                points = utils.removeDuplicatePts(points)
            
            yield points

            i += 1


class BilliardLines:
    def __init__(self, nSides=7, radius=1, cutLength=25, includePolygon=True):
        self.nSides = nSides
        self.radius = radius
        self.cutLength = cutLength
        
        self.verts = utils.polygonVertices(nSides, radius=radius)

        self.sideLength = linalg.norm(self.verts[1] - self.verts[0])

        endPts = []
        for i in range(nSides):
            dir = (self.verts[i] - self.verts[i-1]) / self.sideLength
            endPts.append(self.verts[i] + dir * cutLength)
        self.endPts = np.array(endPts)

        if includePolygon:
            self.cuts = np.array([[self.verts[i-1], endPts[i]] for i in range(nSides)]) #Include central polygon
        else:
            self.cuts = np.array([[self.verts[i], endPts[i]] for i in range(nSides)]) #do not include it
    
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
            j = (i + 1) % self.nSides
            k = (i + 2) % self.nSides

            if includeEdges:
                mask = ((utils.sameSide(self.cuts[i], mid, self.verts[j]) >= 0)
                      & (utils.sameSide(self.cuts[j], mid, self.endPts[i]) >= 0)
                      & ((utils.sameSide(self.cuts[j], mid, self.endPts[i]) > 0)
                       | (utils.sameSide(self.cuts[k], mid, self.verts[i]) > 0)))
            else:
                mask = ((utils.sameSide(self.cuts[i], mid, self.verts[j]) > 0)
                      & (utils.sameSide(self.cuts[j], mid, self.endPts[i]) > 0))
            
            #Must transpose pts to multiply by matrix (and then transpose back)
            rotated = utils.applyMatrixToLines(lines[mask,...] - self.verts[i], M) + self.verts[i]
            newLineSets.append(rotated)

        return np.concatenate(newLineSets)
    
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

        yield lines

        while i < iterations:
            if addPoints and i > 0 and i % addPointsEvery == 0:
                lines = np.concatenate((lines, seedLines))
            
            lines = utils.cutLines(lines, self.cuts, extend=(0, inf))
            lines = self.transformLines(lines)

            if removeDuplicates and i % removeDuplicatesEvery == 0:
                lines = utils.mergeOverlapingSegments(lines)
            
            yield lines

            i += 1


if __name__ == "__main__":
    billiard = BilliardPoints(cutLength=100)
    iterator = billiard.iterator(
        addPoints=False,
        iterations=500,
        seedPoints=5000,
        removeDuplicates=False
    )

    fig = plt.figure()
    initPoints = next(iterator)

    for i, points in enumerate(iterator):
        print(i, len(points))
        # fig.clear()
        # utils.plotPoints(points)
        # plt.pause(0.1)
    
    fig.clear()
    utils.plotPoints(initPoints, color="b")
    utils.plotPoints(points, color="r")
    plt.show()

    initRadius = linalg.norm(initPoints, axis=1)
    finalRadius = linalg.norm(points, axis=1)

    plt.hist(initRadius, 100)
    plt.hist(finalRadius, 100)
    plt.show()

    # print(sum(linalg.norm(P1 - P0) for P1, P0 in initPoints))
    # print(sum(linalg.norm(P1 - P0) for P1, P0 in points))

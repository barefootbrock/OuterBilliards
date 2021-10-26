import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
import utils
from billiards import BilliardPoints, BilliardLines
import time
import imggenerator

iterations = 1000
cutLength = 25

def pointInRegionClosest(self, points, region, includeEdges=True):
    """
    regions are 0 to nSides-1
    Points in the ith region should get reflected accross the ith vertex.
    """
    i, j = region, (region + 1) % self.nSides
    k = (region + 2) % self.nSides

    assert includeEdges
    return (
        (utils.sameSide(self.cuts[i], points, self.verts[j]) >= 0) &
        (utils.sameSide(self.cuts[j], points, self.endPts[i]) >= 0) &
        (
            (utils.sameSide(self.cuts[j], points, self.endPts[i]) > 0) |
            (utils.sameSide(self.cuts[k], points, self.verts[i]) > 0)
        )
    )


def pointInRegionClosest2(self, points, region, includeEdges=True):
    """
    regions are 0 to nSides-1
    Points in the ith region should get reflected accross the ith vertex.
    """
    h = (region - 1) % self.nSides
    i, j = region, (region + 1) % self.nSides

    assert includeEdges
    return (
        (
            (utils.sameSide(self.cuts[i], points, self.verts[j]) > 0) &
            (utils.sameSide(self.cuts[j], points, self.endPts[i]) > 0)
        ) | (
            (utils.sameSide(self.cuts[i], points, self.verts[j]) == 0) &
            (utils.sameSide(self.cuts[h], points, self.verts[i]) > 0)
        )
    )

def pointInRegionFarthest(self, points, region, includeEdges=True):
    """
    regions are 0 to nSides-1
    Points in the ith region should get reflected accross the ith vertex.
    """
    i, j = region, (region + 1) % self.nSides
    k = (region + 2) % self.nSides

    assert includeEdges
    return (
        (utils.sameSide(self.cuts[i], points, self.verts[j]) > 0) &
        (utils.sameSide(self.cuts[j], points, self.endPts[i]) >= 0)
    )


# billiard = BilliardPoints(cutLength=cutLength, includePolygon=False)
# iterator = billiard.iterator(
#     iterations=iterations,
#     addPoints=False
# )
# lines1 = imggenerator.run(iterator, keepAll=True)
# lines1 = utils.mergeOverlapingSegments(lines1)


BilliardLines.pointInRegion = pointInRegionFarthest

billiard = BilliardLines(cutLength=cutLength)
iterator = billiard.iterator(
    iterations=iterations,
    addPoints=False
)
lines2 = imggenerator.run(iterator, keepAll=True)
# lines2 = utils.mergeOverlapingSegments(lines2)

# utils.plotLines(lines1, color="r")
# utils.plotLines(lines2, color="b")
# plt.show()

# print(lines1.shape, lines2.shape)

# print(sum(linalg.norm(P1 - P0) for P1, P0 in lines))
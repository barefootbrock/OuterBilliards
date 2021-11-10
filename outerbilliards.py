import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import utils
from geometry import Region
from transformation import PiecewiseIsometry

class PolygonBillards(PiecewiseIsometry):
    @classmethod
    def regularPolygon(cls, nSides=7, origin=(0, 0), radius=1, singularityLen=25):
        vertices = utils.polygonVertices(nSides, origin, radius)
        return cls(vertices, singularityLen)

    def __init__(self, verts, singularityLen=25):
        """Vertices must be convex and in order.
        singularityLen is the length of the lines extending from vertices"""
        verts = np.asarray(verts, dtype='double')
        isometries, regions = [], []
        mirror = -np.eye(2)
        
        dir = utils.normalize(verts - np.roll(verts, 1, axis=0))
        endPts = dir * singularityLen + verts

        for i, vert in enumerate(verts):
            j = (i + 1) % len(verts)
            isometries.append(utils.matrixAboutPoint(mirror, vert))
            regions.append(Region(
                [endPts[j,:], vert, endPts[i,:]],
                (verts[j,:] + endPts[i,:]) / 2
            ))
            
        super().__init__(isometries, regions)



if __name__ == "__main__":
    from geometry import LineSet
    import matplotlib.pyplot as plt

    B = PolygonBillards.regularPolygon()
    # B = PolygonBillards([
    #     (1, 0),
    #     (-1, 0),
    #     (-1, 1),
    #     (0, 2),
    #     (1, 1)
    # ])
   
    lines = B.singularity().simplify()
    lines.plot()
    plt.show()

    allLines = [lines]

    for i in range(100):
        lines = B(lines).simplify()
        print(i, len(lines))
        allLines.append(lines)
        # lines.plot()
        # plt.show()
    
    LineSet.union(allLines).plot()
    plt.show()
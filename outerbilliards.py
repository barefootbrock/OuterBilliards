import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import utils
from geometry import PointSet, Region
from transformation import PiecewiseIsometry
from params import params

try:
    from scipy.optimize import minimize_scalar
except ImportError:
    print("Scipy not found, SmoothBilliards will not work")


class PolygonBilliards(PiecewiseIsometry):
    REFLECT_BOTH = 'both'
    REFLECT_FAR = 'farthest'
    REFLECT_NONE = 'neither'

    @classmethod
    def regularPolygon(cls, nSides=7, origin=(0, 0), radius=1, **kwargs):
        vertices = utils.polygonVertices(nSides, origin, radius)
        return cls(vertices, **kwargs)

    def __init__(self, verts, singularityLen=25, edgeMethod='both'):
        """Vertices must be convex and in order.
        singularityLen is the length of the lines extending from vertices"""
        verts = np.asarray(verts, dtype=params.dtype)
        isometries, regions = [], []
        mirror = -np.eye(2)
        
        dir = utils.normalize(verts - np.roll(verts, 1, axis=0))
        endPts = dir * singularityLen + verts

        if edgeMethod == 'both':
            includeEdges = [True, True]
        elif edgeMethod == 'farthest':
            includeEdges = [True, False]
        elif edgeMethod == 'neither':
            includeEdges = [False, False]
        else:
            raise ValueError("Invalid edge method!")

        for i, vert in enumerate(verts):
            j = (i + 1) % len(verts)
            isometries.append(utils.matrixAboutPoint(mirror, vert))
            regions.append(Region(
                [endPts[j,:], vert, endPts[i,:]],
                (verts[j,:] + endPts[i,:]) / 2,
                includeEdges
            ))
        
        self.verts = verts
            
        super().__init__(isometries, regions)
    
    def setEdgeMethod(self, edgeMethod):
        if edgeMethod == 'both':
            includeEdges = [True, True]
        elif edgeMethod == 'farthest':
            includeEdges = [True, False]
        elif edgeMethod == 'neither':
            includeEdges = [False, False]
        else:
            raise ValueError("Invalid edge method!")
        
        for region in self.regions:
            region.includeEdges = includeEdges
    
    def plot(self, **kwargs):
        Region(self.verts, (0,0)).plot(**kwargs)


class SmoothBilliards:
    def __init__(self, r):
        """
        Smooth paramitized curve(<f(t), g(t)>)
        """
        def toNpVec(arr):
            arr = np.asarray(arr, dtype=params.dtype)
            return arr.reshape((2, -1)).T
        
        self.r = lambda t: toNpVec(r(t))

        self.center = np.mean(self.r(np.linspace(0, 1)), 0)
    
    def __call__(self, points):
        vert = self.tangentPoint(points)
        newPoints = PointSet(2 * vert - points)
        P1, P2, P3 = points, vert, newPoints
        # print(P1, P2, P3)
        # print(linalg.norm(P1 - P2), linalg.norm(P3 - P2))
        return newPoints
    
    def tangentPoint(self, P):
        """
        Point on curve whose tangent line passes though point
        1: gradient assent to maximize (r(t) - point) . v(t)
        Use Newton's method 5-10 iterations:
            x = x - f(x)/f'(x)
            t = t - ((r(t) - point) X v(t)) ./ ((r(t) - point) X a(t))
        """

        def err(t, point):
            #Error function to minimize
            return np.cross(
                self.center - point,
                utils.normalize(self.r(t) - P)
            )
        
        tVals = []
        for point in PointSet(P):
            t = minimize_scalar(err, args=(point,), tol=1e-12).x % 1
            tVals.append(t)
        
        t = np.asarray(tVals, dtype=params.dtype)

        return self.r(t)

    def plot(self, nPts=100, **kwargs):
        t = np.linspace(0, 1, nPts)
        pts = self.r(t)
        Region(pts, (0, 0)).plot(**kwargs)


if __name__ == "__main__":
    from geometry import LineSet
    import matplotlib.pyplot as plt
    import time

    B = SmoothBilliards(lambda t: [
        4 * cos(2*pi*t),
        7**0.5 * sin(2*pi*t)
    ])

    point = [4, -2.4]
    pts = [point]
    fig = plt.figure()
    plt.xlim(-6, 6)
    plt.ylim(-5, 5)
    B.plot(showEdges=True, color="black")
    plt.pause(5)

    for i in range(10):
        fig.clear()
        plt.xlim(-6, 6)
        plt.ylim(-5, 5)
        B.plot(showEdges=True, color="black")
        PointSet(pts).plot(size=25)
        lines = LineSet.connect(pts)
        lines.plot(size=1)
        plt.pause(1)
        
        point = B(point)
        print(point)
        pts.append(point[0,:])

    plt.show()

    # B = PolygonBillards.regularPolygon()
   
    # lines = B.singularity().simplify()
    # lines.plot()
    # plt.show()

    # allLines = [lines]
    # t = time.time()
    # for i in range(1000):
    #     lines = B(lines).simplify()
    #     if i % 100 == 0: print(i, len(lines))
    #     allLines.append(lines)
    #     # lines.plot()
    #     # plt.show()
    # print(time.time() - t)
    # LineSet.union(*allLines).plot()
    # plt.show()
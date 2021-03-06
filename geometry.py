import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
from numpy.core.function_base import linspace
import utils
import matplotlib.pyplot as plt
from params import params

class PointSet(np.ndarray):
    def union(*args, simplfy=True):
        sets = []
        for arg in args:
            if not isinstance(arg, PointSet):
                arg = PointSet(arg)
            sets.append(arg)
        
        pts = np.concatenate(sets)

        if simplfy:
            pts = utils.removeDuplicatePts(pts)
        
        return PointSet(pts)
    
    def __new__(cls, points=None):
        if points is None:
            points = np.zeros((0, 2), dtype=params.dtype)
        arr = np.asarray(points, dtype=params.dtype)

        if arr.shape == (2,): #Single point
            arr = arr.reshape((1, 2))
        
        n = arr.shape[0]
        assert arr.shape == (n, 2), "Invalid point set shape!"
        
        return arr.view(cls)
    
    def __or__(self, other):
        """Union"""
        return self.union(other, simplfy=False)
    
    def __and__(self, other):
        """Intersection"""
        raise NotImplementedError()
    
    def __xor__(self, other):
        """Union - intersection"""
        raise NotImplementedError()
    
    def within(self, region):
        """Get all points in region"""
        mask = region.test(self)
        return self[mask,:]
    
    def transform(self, mat):
        """Apply matrix (2x2 or 3x3)"""
        mat = np.asarray(mat, params.dtype)

        if mat.shape == (2, 2):
            pts = (mat @ self.T).T
            return type(self)(pts)
        
        elif mat.shape == (3, 3):
            #Apply matrix to homogenious coordinates
            pts = (mat[:2,:2] @ self.T).T + mat[:2,2]
            return type(self)(pts)
        
        else:
            raise ValueError("Transformation must be a 2x2 or 3x3 matrix")
    
    def simplify(self):
        pts = utils.removeDuplicatePts(self)
        return type(self)(pts)
    
    def memory(self):
        return self.nbytes

    def plot(self, *args, **kwargs):
        return utils.plotPoints(self, *args, **kwargs)


class LineSet(np.ndarray):
    @classmethod
    def connect(cls, verts):
        verts = PointSet(verts)
        P0, P1 = verts[:-1,:], verts[1:,:]
        lines = np.stack((P0, P1), axis=1)
        return cls(lines)
    
    def union(*args, simplfy=True):
        sets = []
        for arg in args:
            if not isinstance(arg, LineSet):
                arg = LineSet(arg)
            sets.append(arg)

        lines = np.concatenate(sets)

        if simplfy:
            lines = utils.mergeOverlapingSegments(lines)
        return LineSet(lines)
    
    def __new__(cls, lines=None):
        if lines is None:
            arr = np.zeros((0, 2, 2))
        else:
            arr = np.asarray(lines, dtype=params.dtype)

        if arr.shape == (2, 2): #Single line
            arr = arr.reshape((1, 2, 2))
        
        n = arr.shape[0]
        assert arr.shape == (n, 2, 2), "Invalid line set shape!"
        
        return arr.view(cls)
    
    def __or__(self, other):
        """Union"""
        return self.union(other, simplfy=False)
    
    def __and__(self, other):
        """Intersection"""
        raise NotImplementedError()
    
    def __xor__(self, other):
        """Union - intersection"""
        raise NotImplementedError()

    def midpoint(self):
        return PointSet(utils.midpoint(self))

    def within(self, region, cut=True):
        """Get line segments within the region, splitting if nessary"""
        if cut:
            assert region.convex
            #Extend cut lines infinitly. Region is convex, so this works
            cut = self.cut(region.boundryEdges(), extend=(-inf, inf))
            return cut.within(region, cut=False)
        else:
            return self[region.test(self.midpoint()),:,:]
    
    def cut(self, cuts, extend=(0, 1)):
        """Split lines that intersect a cut"""
        lines = utils.cutLines(self, cuts, extend=extend)
        return type(self)(lines)
    
    def transform(self, mat):
        """Apply matrix (2x2 or 3x3)"""
        mat = np.asarray(mat, params.dtype)

        if mat.shape == (2, 2):
            res = self.copy()
            res[:,0,:] = (mat @ self[:,0,:].T).T
            res[:,1,:] = (mat @ self[:,1,:].T).T
            return res
        
        elif mat.shape == (3, 3):
            #Apply matrix to homogenious coordinates
            res = self.copy()
            res[:,0,:] = (mat[:2,:2] @ self[:,0,:].T).T + mat[:2,2]
            res[:,1,:] = (mat[:2,:2] @ self[:,1,:].T).T + mat[:2,2]
            return res
        
        else:
            raise ValueError("Transformation must be a 2x2 or 3x3 matrix")
    
    def simplify(self):
        lines = utils.mergeOverlapingSegments(self)
        return type(self)(lines)

    def totalLen(self):
        return np.sum(linalg.norm(self[:,0,:] - self[:,1,:], axis=1))
    
    def pointSpread(self, nPoints=100):
        """Evenly distribute points along lines"""
        points = PointSet()
        totalLen = self.totalLen()
        lenUsed = 0
        ptsUsed = 0

        for P0, P1 in self:
            lenUsed += linalg.norm(P1 - P0) / totalLen
            count = round(nPoints * lenUsed) - ptsUsed
            ptsUsed += count

            t = np.linspace(0, 1, count)
            newPts = t[:,np.newaxis] * (P1 - P0) + P0
            
            points = points | newPts
        
        return points
    
    def memory(self):
        return self.nbytes
    
    def plot(self, *args, **kwargs):
        return utils.plotLines(self, *args, **kwargs)


class Region:
    convex = True

    def __init__(self, vertices, interiorPoint, includeEdges=True):
        """
        A convex region in the plane bounded by a polygon.
        If region is not closed, first/last edges will be extended.
        Each edge can, optionally, be included in the region.
        """
        self.verts = PointSet(vertices)
        self.interiorPoint = np.asarray(interiorPoint, dtype=params.dtype)

        if np.isscalar(includeEdges):
            self.includeEdges = np.ones(len(self.verts) - 1, 'bool')
            self.includeEdges *= bool(includeEdges)
        else:
            self.includeEdges = np.asarray(includeEdges, 'bool')
        
        assert len(self.verts) > 1
        assert len(self.includeEdges) == len(self.verts) - 1
   
    def test(self, points, overrideInclude=None):
        """Test if point(s) are in the region"""
        res = np.ones(len(points), 'bool')

        for i in range(len(self.verts) - 1):
            edge = [self.verts[i], self.verts[i+1]]
            include = self.includeEdges[i]
            if overrideInclude is not None:
                include = overrideInclude
            
            if include:
                extend1 = i == 0 #Extend first line
                extend2 = i + 2 == len(self.verts) #Extend last line
                res &= ((utils.sameSide(edge, points, self.interiorPoint) > 0)
                      | (utils.onLine(edge, points, extend1, extend2)))
            else:
                res &= utils.sameSide(edge, points, self.interiorPoint) > 0

        return res
    
    def boundry(self):
        """Region boundry (as verts of polygon)"""
        return self.verts
    
    def boundryEdges(self):
        """Boundry as a set of line segments"""
        return LineSet.connect(self.verts)
    
    def plot(self, showEdges=False, alpha=0.3, **kwargs):
        plt.fill(self.verts[:,0], self.verts[:,1], alpha=alpha, **kwargs)
        if showEdges:
            self.boundryEdges().plot(**kwargs)


def iterate(obj, f, n, history=False):
    """f(f(...f(obj)...)) n times"""
    hist = [obj]
    for i in range(n):
        obj = f(obj)
        if history:
            hist.append(obj)
    
    if history:
        return obj, hist
    else:
        return obj
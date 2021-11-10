import numpy as np
from numpy import array, linalg
from numpy import sin, cos, tan, pi, inf
import utils
import matplotlib.pyplot as plt


class PointSet(np.ndarray):
    def __new__(cls, points):
        arr = np.asarray(points, dtype='double')

        if arr.shape == (2,): #Single point
            arr = arr.reshape((1, 2))
        
        n = arr.shape[0]
        assert arr.shape == (n, 2), "Invalid point set shape!"
        
        return arr.view(cls)
    
    def __or__(self, other):
        """Union"""
        pts = utils.removeDuplicatePts(np.concatenate((self, other)))
        return type(self)(pts)
    
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
        mat = np.asarray(mat, 'double')
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

    def plot(self, *args, **kwargs):
        utils.plotPoints(self, *args, **kwargs)


class LineSet(np.ndarray):
    @classmethod
    def connect(cls, verts):
        verts = PointSet(verts)
        P0, P1 = verts[:-1,:], verts[1:,:]
        lines = np.stack((P0, P1), axis=1)

        return cls(lines)
    
    def __new__(cls, lines):
        arr = np.asarray(lines, dtype='double')

        if arr.shape == (2, 2): #Single line
            arr = arr.reshape((1, 2, 2))
        
        n = arr.shape[0]
        assert arr.shape == (n, 2, 2), "Invalid line set shape!"
        
        return arr.view(cls)
    
    def __or__(self, other):
        """Union"""
        raise NotImplementedError()
    
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
        mat = np.asarray(mat, 'double')
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
    
    def plot(self, *args, **kwargs):
        utils.plotLines(self, *args, **kwargs)


class Region:
    convex = True

    def __init__(self, vertices, interiorPoint, includeEdges=True):
        """
        A convex region in the plane bounded by a polygon.
        If region is not closed, first/last edges will be extended.
        Each edge can, optionally, be included in the region.
        """
        self.verts = PointSet(vertices)
        self.interiorPoint = np.asarray(interiorPoint, dtype='double')

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
                extend2 = i + 2 == len(self.verts) #Extend first line
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
    
    def plot(self, **kwargs):
        plt.fill(self.verts[:,0], self.verts[:,1], **kwargs)
    
    # def splitOnBoundry(self, lines):
    #     """Split a set of line segments so that none cross the boundry"""
    #     #Must filter split points so they are not outside region
    #     return utils.cutLines(
    #         lines,
    #         self.edges,
    #         extend=(-inf, inf),
    #         pointFilter=lambda p: self.test(p, overrideInclude=True)
    #     )


class PiecewiseIsometry:
    def __init__(self, mats, regions):
        """Mats are matrices that define isometries"""
        self.mats = mats
        self.regions = regions
   
    def __call__(self, obj):
        results = []
        for mat, region in zip(self.mats, self.regions):
            results.append(obj.within(region).transform(mat))
        
        return np.concatenate(results).view(type(obj))
    
    def singularity(self, simplify=True):
        """Line segments on boundries of regions"""
        lines = [r.boundryEdges() for r in self.regions]

        lines = LineSet(np.concatenate(lines))
        if simplify:
            lines = lines.simplify()
        
        return lines
    
    # def splitOnBoundries(self, lines):
    #     return reduce(lambda l, r: r.splitOnBoundry(l), self.regions, lines)


class PolygonOuterBillards(PiecewiseIsometry):
    def __init__(self, verts, singularityLen=25):
        """Vertices must be convex and in order.
        singularityLen is the length of the lines extending from vertices"""
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


class RegularPolygonOuterBillards(PolygonOuterBillards):
    def __init__(self, nSides=7, origin=(0, 0), radius=1, singularityLen=25):
        vertices = utils.polygonVertices(nSides, origin, radius)
        super().__init__(vertices, singularityLen)

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


if __name__ == "__main__":
    B = RegularPolygonOuterBillards(singularityLen=25)
   
    lines = B.singularity()

    lines.plot()
    plt.show()

    # fig = plt.figure()
    allLines = [lines]

    for i in range(300):
        lines = B(lines).simplify()
        print(i, len(lines))
        # lines.plot()
        # plt.show()
        allLines.append(lines)
    
    # plt.show()

    # lines.plot()
    LineSet(np.concatenate(allLines)).simplify().plot()
    plt.show()


    # A = np.array([0.5, 0.5*tan(2*pi/5)])
    # B = np.array([0.0, 0.0])
    # C = np.array([1.0, 0.0])
    # D = np.array([cos(pi/5), sin(pi/5)])

    # S0 = np.array([0.5, 0.5*tan(3*pi/10)])
    # S1 = np.array([0.5, 0.5*tan(pi/10)])

    # T0 = np.array([[+cos(4*pi/5), +sin(4*pi/5)],
    #                [-sin(4*pi/5), +cos(4*pi/5)]])
    # T0 = utils.matrixAboutPoint(T0, S0)

    # T1 = np.array([[+cos(4*pi/5), -sin(4*pi/5)],
    #                [+sin(4*pi/5), +cos(4*pi/5)]])
    # T1 = utils.matrixAboutPoint(T1, S1)

    # R0 = Region([A, B, D, A], S0, [True, False, True])
    
    # R1 = Region([B, C, D, B], S1, [True, True, True])

    # iso = PiecewiseIsometry((T0, T1), (R0, R1))

    # plt.axis("equal")

    # lines = iso.singularity()
    # lines.plot()
    # plt.show()

    # fig = plt.figure()

    # allLines = [lines]

    # for i in range(5000):
    #     lines = iso(lines).simplify()
    #     # print(i, len(lines), lines.totalLen())
    #     # fig.clear()
    #     # lines.plot()
    #     # plt.pause(0.05)
    #     allLines.append(lines)
    
    # plt.show()

    # LineSet(np.concatenate(allLines)).simplify().plot()
    # plt.show()
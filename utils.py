import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
from numpy.linalg.linalg import norm
from params import params
from functools import cmp_to_key

eps = params.eps


def sameSide(line, point, refPoint):
    """
    Test if a point is on the same side of a line as refPoint
    Returns 1 if same side, -1 if other side, and 0 if on the line
    line has shape (_, 2, 2)
    point & refPoint have shape (_, 2)
    eps is the minimum distance to the line to be on the line
    """
    line = np.reshape(line, (-1, 2, 2)).astype(np.float64)
    point = np.reshape(point, (-1, 2)).astype(np.float64)
    refPoint = np.reshape(refPoint, (-1, 2)).astype(np.float64)
    
    P0, P1 = line[...,0,:], line[...,1,:]
    direction = P1 - P0
    cross1 = np.cross(point - P0, direction)
    cross2 = np.cross(refPoint - P0, direction)
    cross1[np.abs(cross1) * linalg.norm(direction) < eps] = 0
    return np.sign(cross1) * np.sign(cross2)

def onLine(line, point, extend1=False, extend2=False):
    """
    Test if point is on the line segment
    extend1 extends line past first point (same with extend2)
    eps is the minimum distance to the line
    """
    line = np.reshape(line, (-1, 2, 2)).astype(np.float64)
    point = np.reshape(point, (-1, 2)).astype(np.float64)
    
    P0, P1 = line[...,0,:], line[...,1,:]
    direction = P1 - P0
    cross = np.cross(point - P0, P1 - P0)
    
    dot1 = np.sum((point - P0) * (P1 - P0), axis=1) if not extend1 else 1
    dot2 = np.sum((point - P1) * (P0 - P1), axis=1) if not extend2 else 1
    
    return ((np.abs(cross) < eps * linalg.norm(direction))
          & (dot1 * dot2 > -eps * linalg.norm(direction)**2))

def midpoint(line):
    """
    Return midpoint(s) lines(s)
    """
    return (line[...,0,:] + line[...,1,:]) / 2

def applyMatrixToLines(lines, M):
    """
    Apply a matrix to a set of lines
    """
    res = np.empty_like(lines)
    res[:,0,:] = (M @ lines[:,0,:].T).T #Apply to 1st point
    res[:,1,:] = (M @ lines[:,1,:].T).T #2nd point
    return res

def parallel(line1, line2):
    """
    Test if lines are parallel
    line1 & line2 have shape (_, 2, 2)
    result has shape (_,)
    eps is roughly the max angle between lines
    """
    line1 = np.reshape(line1, (-1, 2, 2)).astype(np.float64)
    line2 = np.reshape(line2, (-1, 2, 2)).astype(np.float64)

    #Put lines in parametric form:
    # <p0,p1> + <u0,u1>s, <q0,q1> + <v0,v1>t
    P, Q = line1[...,0,:], line2[...,0,:]
    U, V = line1[...,1,:] - P, line2[...,1,:] - Q
    return np.abs(np.cross(U, V)) < eps * linalg.norm(U) * linalg.norm(V)

def intersect(line1, line2, sBounds=(-eps, 1+eps), tBounds=(-eps, 1+eps)):
    """
    Find intersection point of two line segments
    Returns if they intersect and where
    line1 & line2 have shape (_, 2, 2)
    sBounds & tBounds spesify how to extend line segments
        ((0, 1) does not extend, (0, inf) extents toward second point, etc.)
    result has shape (_,), (_, 2)
    """
    line1, line2 = np.broadcast_arrays(line1, line2)
    #one line or a set of lines at once
    multi = line1.ndim > 2

    line1 = np.reshape(line1, (-1, 2, 2)).astype(np.float64)
    line2 = np.reshape(line2, (-1, 2, 2)).astype(np.float64)

    assert line1.shape == line2.shape

    mask = ~parallel(line1, line2)

    #Put lines in parametric form:
    # <p0,p1> + <u0,u1>s, <q0,q1> + <v0,v1>t
    P, Q = line1[...,0,:], line2[...,0,:]
    U, V = line1[...,1,:] - P, line2[...,1,:] - Q

    #Put in matrix form and solve Ax=B where
    # A = [U, -V]
    # x = [s; t]
    # B = Q - P
    A = np.empty_like(line1)
    A[...,:,0] = U
    A[...,:,1] = -V
    B = Q - P

    x = np.zeros_like(U)
    x[mask,...] = linalg.solve(A[mask,...], B[mask,...])
    s, t = x[...,0], x[...,1]

    # print("P", P, "Q", Q)
    # print("U", U, "V", V)
    # print("A", A)
    # print("B", B)
    # print("x", x)
    # print("s", s, "t", t)
    # print("P1", (P + U*s)[mask,...], "P2", (Q + V*t)[mask,...])
    
    #Get intersection point
    point = P + U*s[:,np.newaxis]
    # samePoint = Q + V*t[:,np.newaxis] #Should be exactly the same
    # assert np.allclose(point[mask,...], samePoint[mask,...])

    #Lines must intersect between end points (0 < s < 1)
    sMin, sMax = sBounds
    tMin, tMax = tBounds
    valid = (sMin <= s) & (s <= sMax) & (tMin <= t) & (t <= tMax)
    
    mask[~valid] = False

    #If a single value was given only return a single value
    if multi:
        return mask, point
    else:
        return mask[0], point[0,...]

def cutLines(lines, cuts, extend=(0, 1), pointFilter=None):
    """
    Cut a set of line segments with a set of cuts line segments.
    Cuts can optionally be extended toward either side.
    Intersection points can, optionally, be filtered with pointFilter function.
    """
    lines = np.reshape(lines, (-1, 2, 2)).astype(np.float64)
    cuts = np.reshape(cuts, (-1, 2, 2)).astype(np.float64)
    #Do not allow line to be cut at end point (this would leave a 0 length line)
    sBounds = eps, 1-eps
    #Allow line to be cut with end point of cut line
    tBounds = extend[0] - eps, extend[1] + eps
    
    for cut in cuts:
        wasCut, cutPts = intersect(lines, cut,
                                   sBounds=sBounds, tBounds=tBounds)
        
        if pointFilter is not None:
            wasCut[wasCut] &= pointFilter(cutPts[wasCut,:])

        notCut = lines[~wasCut,...]
        segment1 = np.stack((lines[wasCut,0,:], cutPts[wasCut,:]), axis=1)
        segment2 = np.stack((cutPts[wasCut,:], lines[wasCut,1,:]), axis=1)
        lines = np.concatenate((notCut, segment1, segment2))
    
    return lines

def polygonVertices(sides, center=(0, 0), radius=1, sideLength=None):
    """
    Calculate the vertices of a regular polygon.
    Radius defaults to 1 if not spesified.
    Radius and sideLength cannot both be provided.
    """
    if sideLength is not None:
        radius = sideLength / 2 / sin(pi / sides)
    
    verts = [np.array([0, radius])]
    vertRotation = np.array([[+cos(2*pi/sides), +sin(2*pi/sides)],
                             [-sin(2*pi/sides), +cos(2*pi/sides)]])
    for i in range(sides-1):
        verts.append(vertRotation @ verts[-1])

    return np.array(verts) + center

def removeDuplicatePts(points):
    """
    Remove duplicate points
    """
    points = np.asarray(points, dtype=params.dtype)
    decimals = -int(np.log10(eps))
    _, idx = np.unique(np.round(points, decimals), axis=0, return_index=True)
    return points[idx,:]

def removeDuplicatePts2(points):
    """
    Remove duplicate points using improved (?) algorithm
    """
    def cmpFunc(A, B):
        if (A[0] - B[0])**2 + (A[1] - B[1])**2 < eps**2:
            return 0
        if A[0] != B[0]:
            return A[0] - B[0]
        return A[1] - B[1]

    points = sorted(points, key=cmp_to_key(cmpFunc))
    unique = [points[0]]
    for P in points[1:]:
        if cmpFunc(P, unique[-1]) != 0:
            unique.append(P)

    return np.asarray(unique, dtype=params.dtype)

def mergeOverlapingSegments(lines):
    """
    Given a set of line segments, returns a smaller set that looks the same.
    If 2 line segments are colinear and overlap, they are joined into one.
    """
    lines = np.reshape(lines, (-1, 2, 2)).astype(np.float64)
    
    def merge(lines):
        """Sub-function for nonvertical lines."""
        if len(lines) < 2:
            return lines

        #Put lines in parametric form and pointing right:
        # <p0,p1> + <u0,u1>s
        leftPtIdx = np.argmin(lines[:,:,0], axis=1)
        idxs = np.arange(len(lines))
        P = lines[idxs,leftPtIdx,:]
        Q = lines[idxs,1-leftPtIdx,:]
        U = Q - P
        
        # print("lines", lines)
        # print("P", P)
        # print("U", U)

        #sort by slope, then y-intercept, then x coord of first point
        decimals = -int(np.log10(eps))
        m = np.round(U[:,1] / U[:,0], decimals)
        b = np.round(P[:,1] - (U[:,1] / U[:,0]) * P[:,0], decimals)
        x0 = P[:,0]

        # print("m", m)
        # print("b", b)
        # print("x0", x0)

        order = np.lexsort((x0, b, m)) #last elements are sorted on first

        # print("order", order)

        newLines = []
        firstPoint = P[order[0],:]
        lastPoint = Q[order[0],:]
        currM = m[order[0]]
        currB = b[order[0]]

        # print("fP", firstPoint)
        # print("lP", lastPoint)
        # print("curr M, B", currM, currB)

        for i in order[1:]:
            # print("P0, P1, m, b", P0, P1, m[i], b[i])

            if (currM == m[i] and
                currB == b[i] and
                P[i,0] <= lastPoint[0] + eps):
                #Lines overlap and can be joined
                if Q[i,0] > lastPoint[0] + eps: #Line extends past end point
                    lastPoint = Q[i,:]
            else:
                newLines.append([firstPoint, lastPoint])
                firstPoint = P[i,:]
                lastPoint = Q[i,:]
                currM, currB = m[i], b[i]
        
        newLines.append([firstPoint, lastPoint])

        return np.array(newLines)
    
    length = linalg.norm(lines[:,1,:] - lines[:,0,:], axis=1)
    lines = lines[length > eps,...]

    steep = np.abs(lines[:,1,1] - lines[:,0,1]) > np.abs(lines[:,1,0] - lines[:,0,0])
    return np.concatenate((
        merge(lines[~steep,:,:]),
        merge(lines[steep,:,::-1])[:,:,::-1] #swap x and y
    ))
    
def matrixAboutPoint(M2x2, center):
    """Convert 2x2 matrix aroung a point to 3x3"""
    M = np.eye(3, dtype=params.dtype)
    center = np.asarray(center)
    M[:2,:2] = M2x2
    M[:2,2] = center - (M2x2 @ center[:,np.newaxis])[:,0]
    return M

def normalize(vecs):
    """Scale vecs to have length of 1"""
    return vecs / linalg.norm(vecs, axis=1)[:,np.newaxis]


def plotPoints(points, size=0.4, color="r", plot=plt, setAspect=True):
    """
    Plot a set of points with plt
    plt.show or plt.pause must be called to see the plot
    """
    if setAspect:
        plot.gca().set_aspect('equal', adjustable='box')

    return plot.scatter(points[...,0], points[...,1], s=size, c=color, marker=".")

def plotLines(lines, size=0.4, color="b", showPoints=False, pointSize=None, pointColor="r",
              plot=plt, setAspect=True):
    """
    Plot a set of line segments (and optionally theit end points) with plt
    plt.show or plt.pause must be called to see the plot
    """
    lines = np.reshape(lines, (-1, 2, 2)).astype(np.float64)

    coords = np.empty((lines.shape[0] * 3, 2))

    coords[::3,:] = lines[:,0,:]
    coords[1::3,:] = lines[:,1,:]
    coords[2::3,:] = np.nan

    if setAspect:
        plot.gca().set_aspect('equal', adjustable='box')

    if showPoints:
        if pointSize == None: pointSize = 10 * size**2

        plot.scatter(lines[...,0,0], lines[...,0,1], s=pointSize, c=pointColor, marker=".")
        plot.scatter(lines[...,1,0], lines[...,1,1], s=pointSize, c=pointColor, marker=".")
    
    return plot.plot(coords[:,0], coords[:,1], color, linewidth=size)

if __name__ == "__main__":
    # lines = [[[0, 1], [1, 0]],
    #          [[0, 0], [-1, -1]]]
    
    # cut = [[-0.75, -1], [1.25, 1]]
    # plotLines(lines, showPoints=True)
    # plotLines(cut, color="g", showPoints=True)
    # plt.show()

    # print(intersect(lines, [cut, cut]))

    # lines = cutLines(lines, cut)
    # plotLines(lines, showPoints=True)
    # plotLines(cut, color="g", showPoints=True)
    # plt.show()

    # plotPoints(polygonVertices(70))
    # plt.show()

    # print(lines)

    # lines = [[[0, 0], [2, 2]],
    #          [[3, 3], [1, 1]]]
    # merge = mergeOverlapingSegments(lines)
    # print(merge)

    # plotLines(lines, size=1, showPoints=True)
    # plt.show()
    # plotLines(merge, size=1, showPoints=True)
    # plt.show()

    # print(onLine(lines, [-1, -1], extend2=True))

    # print(onLine([[0, 0], [1, 1]], [(0, 0), (1.00000001, 1.00000001)]))

    print(removeDuplicatePts2([
        [0.4999, 0.6249],
        [0.5001, 0.6051]
    ]))
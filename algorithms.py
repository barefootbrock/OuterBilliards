from copy import deepcopy
from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt
import time

#Different algorithms for generating singularity structure

def usingPoints(billiard, iterations, seedPoints=1000,
                keepPrev=True, simplify=True, verbose=True, trackMemory=False):
    if verbose:
        print("Running points with %i iterations, seedPoints=%i, keepPrev=%i, simplify=%i" %(
        iterations, seedPoints, keepPrev, simplify))
    
    singularity = billiard.singularity().simplify()
    points = singularity.pointSpread(seedPoints)

    if keepPrev:
        allPoints = [points]
    
    t0 = time.time()

    for i in range(iterations):
        points = billiard(points)

        if simplify:
            points = points.simplify()
        if keepPrev:
            allPoints.append(points)
        
        if verbose and i * 10 // iterations != (i + 1) * 10 // iterations:
            print("%i: %i points" % (i + 1, len(points)))

    if keepPrev:
        if trackMemory:
            pointCounts = [len(p) for p in allPoints]
        points = PointSet.union(*allPoints, simplfy=False)
    
    finalMemory = points.memory()
    points = points.simplify()

    if verbose:
        print("Done (%i final points)" % len(points))
        print("Run time: %.3gs" % (time.time() - t0))
        print("Memory of final (unsimplified) result: %.3gMB" % (finalMemory / 1024**2))

    if keepPrev and trackMemory:
        return points, pointCounts
    return points


def usingLines(billiard, iterations, keepPrev=True, simplify=True,
               edgeMethod='both', useSymmetry=False, verbose=True, trackMemory=False):
    """
    singularityLen: length of singularity to use
    keepPrev: keep all iterations
    simplify: remove overlapping lines every iteration
    splitMode:
        'both' = split lines landing on singulatity
        'farthest' = Only use farthest vertex
        'neither' = Remove line segments landing on singularity after 1st iteration
    useSummetry: Use rotational symmetry to speed up
    """
    billiard = deepcopy(billiard) #So changes will not affect original billiard

    if verbose:
        print("Running Lines with %i iterations, keepPrev=%i, simplify=%i, edgeMethod=%s, useSymmetry=%i" %(
        iterations, keepPrev, simplify, edgeMethod, useSymmetry))
    
    lines = billiard.singularity().simplify()

    if useSymmetry:
        lines = lines[0:1,:,:]

    if keepPrev:
        allLines = [lines]

    t0 = time.time()

    for i in range(iterations):
        if i == 1:
            billiard.setEdgeMethod(edgeMethod)
        
        lines = billiard(lines)

        if simplify:
            lines = lines.simplify()
        if keepPrev:
            allLines.append(lines)

        if verbose and i * 10 // iterations != (i + 1) * 10 // iterations:
            print("%i: %i lines" % (i + 1, len(lines)))

    if keepPrev:
        if trackMemory:
            lineCounts = [len(l) for l in allLines]
        lines = LineSet.union(*allLines, simplfy=False)
    
    finalMemory = lines.memory()
    if keepPrev:
        lines = lines.simplify()
    
    if useSymmetry:
        lines = utils.applySymmetry(lines, rotational=len(billiard.verts))
        if finalMemory < lines.memory():
            print("Memory of simplified result > memory of single piece.")
            finalMemory = lines.memory()

    if verbose:
        print("Done (%i final line segments)" % len(lines))
        print("Run time: %.3gs" % (time.time() - t0))
        print("Memory of final (unsimplified) result: %.3gMB" % (finalMemory / 1024**2))
    
    if keepPrev and trackMemory:
        return lines, lineCounts
    return lines


if __name__ == "__main__":
    B = PolygonBilliards.regularPolygon()
    usingPoints(B, 100, seedPoints=1400).plot()
    usingLines(B, 100).plot()
    B.plot()
    plt.show()
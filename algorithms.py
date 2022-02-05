from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt
import time

#Different algorithms for generating singularity structure

def usingPoints(billiard, iterations, seedPoints=1000,
                keepPrev=True, simplify=True, verbose=True):
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
        points = PointSet.union(*allPoints, simplfy=False)
    
    finalMemory = points.memory()
    points = points.simplify()

    if verbose:
        print("Done (%i final points)" % len(points))
        print("Run time: %.3gs" % (time.time() - t0))
        print("Memory of final (unsimplified) result: %.3gMB" % (finalMemory / 1024**2))
    
    return points


def usingLines(billiard, iterations, keepPrev=True, simplify=True,
               edgeMethod='both', useSymmetry=False, verbose=True):
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
        lines = LineSet.union(*allLines, simplfy=False)
    
    finalMemory = lines.memory()
    if keepPrev:
        lines = lines.simplify()
    
    if useSymmetry:
        utils.applySymmetry(lines, rotational=len(billiard.verts))

        lines = LineSet.union(*allLines)

    if verbose:
        print("Done (%i final line segments)" % len(lines))
        print("Run time: %.3gs" % (time.time() - t0))
        print("Memory of final (unsimplified) result: %.3gMB" % (finalMemory / 1024**2))
    
    return lines


if __name__ == "__main__":
    usingPoints(200, seedPoints=1400, singularityLen=25).plot()
    usingLines(200, singularityLen=25).plot()
    PolygonBilliards.regularPolygon().plot()
    plt.show()
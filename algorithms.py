from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt
import time

#Different algorithms for generating singularity structure

def usingPoints(iterations, sides=7, singularityLen=25, seedPoints=1000,
                keepPrev=True, simplify=True, verbose=True):
    isometry = PolygonBillards.regularPolygon(
        nSides=sides,
        singularityLen=singularityLen
    )
    singularity = isometry.singularity().simplify()
    points = singularity.pointSpread(seedPoints)

    if keepPrev:
        allPoints = [points]
    
    t0 = time.time()

    for i in range(iterations):
        points = isometry(points)
        if simplify:
            points = points.simplify()
        if keepPrev:
            allPoints.append(points)
        if verbose:
            print("%i: %i points" % (i, len(points)))

    if keepPrev:
        points = PointSet.union(allPoints)
    if verbose:
        print("Done (%i final points)" % len(points))
        print("Run time: %.2fs" % (time.time() - t0))
    return points


def usingLines(iterations, sides=7, singularityLen=25, keepPrev=True,
               simplify=True, splitMode='split', useSymmetry=False, verbose=True):
    """
    singularityLen: length of singularity to use
    keepPrev: keep all iterations
    simplify: remove overlapping lines every iteration
    splitMode:
        'split' = split lines landing on singulatity
        'farthest' = Only use farthest vertex
        'remove' = Remove line segments landing on singularity after 1st iteration
    useSummetry: Use rotational symmetry to speed up
    """

    if splitMode == 'split':
        isometry = PolygonBillards.regularPolygon(
            nSides=sides,
            singularityLen=singularityLen)
    elif splitMode in ('farthest', 'remove'):
        isometry = PolygonBillards.regularPolygon(
            nSides=sides,
            singularityLen=singularityLen,
            edgeMethod="farthest"
        )
    else:
        raise ValueError("Invalid split mode")
    
    lines = isometry.singularity().simplify()

    if useSymmetry:
        lines = lines[0:1,:,:]

    if keepPrev:
        allLines = [lines]

    t0 = time.time()

    for i in range(iterations):
        if splitMode == 'remove' and i == 1:
            #after first iteration ignore points on singularities
            for r in isometry.regions:
                r.includeEdges[:] = False
            simplify = False
        
        lines = isometry(lines)

        if simplify:
            lines = lines.simplify()
        if keepPrev:
            allLines.append(lines)
        if verbose:
            print("%i: %i lines %.3f" % (i, len(lines), lines.totalLen()))

    if keepPrev:
        lines = LineSet.union(allLines)
    
    if useSymmetry:
        M = [[cos(2*pi/sides), -sin(2*pi/sides)],
             [sin(2*pi/sides),  cos(2*pi/sides)]]
        
        allLines = [lines]
        for i in range(sides - 1):
            lines = lines.transform(M)
            allLines.append(lines)

        lines = LineSet.union(allLines)

    if verbose:
        print("Done (%i final line segments)" % len(lines))
        print("Run time: %.2fs" % (time.time() - t0))
    
    return lines


if __name__ == "__main__":
    usingPoints(300, seedPoints=7, singularityLen=5, keepPrev=False).plot(size=2)
    # usingLines(6, singularityLen=5, keepPrev=False).plot(size=1)
    PolygonBillards.regularPolygon().plot()
    plt.show()
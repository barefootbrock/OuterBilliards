from numpy import sign
from algorithms import usingLines, usingPoints
from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt
import time

#code for all the figures in the presentation

def ovalBillardAnim():
    oval = lambda t: [2*cos(2*pi*t), sin(2*pi*t)]
    B = SmoothBilliards(oval)

    fig = plt.figure()
    plt.xlim(-5, 5)
    plt.ylim(-3, 3)
    # plt.axes("equal")

    point = PointSet([1, 2.3])
    pts = [point[0,:]]

    B.plot(showEdges=True, color="black")
    point.plot(size=100)
    plt.pause(10)

    for i in range(6):
        point = B(point)
        pts.append(point[0,:])
        
        fig.clear()
        B.plot(showEdges=True, color="black")
        lines = LineSet.connect(pts)
        lines.plot(showPoints=True, pointSize=100, size=1)
        plt.xlim(-5, 5)
        plt.ylim(-3, 3)
        plt.pause(2)
    
    plt.show()

def pentagonSinglePointAnim():
    B = PolygonBilliards.regularPolygon(nSides=5)

    fig = plt.figure()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    point = PointSet([0.9, 1.2])
    pts = [point[0,:]]

    B.plot(showEdges=True, color="black")
    point.plot(size=100)
    plt.pause(10)

    for i in range(6):
        point = B(point)
        pts.append(point[0,:])
        
        fig.clear()
        B.plot(showEdges=True, color="black")
        lines = LineSet.connect(pts)
        lines.plot(showPoints=True, pointSize=100, size=1)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.pause(2)
    
    plt.show()

def piecewiseIsometryVisual():
    B = PolygonBilliards.regularPolygon(nSides=5)

    fig = plt.figure()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    B.plot(color="black")
    B.singularity().plot(size=1)
    B.regions[0].plot()
    plt.show()

def dualValuedVisual():
    B = PolygonBilliards.regularPolygon(nSides=5)

    fig = plt.figure()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    pts = PointSet([
        2.5 * cos(pi/5),
        1 - 2.5 * sin(pi/5)
    ])

    B.plot(showEdges=True, color="black")
    B.singularity().plot(size=1)
    pts.plot(size=100)
    B(pts).plot(size=100)
    plt.show()

def pentagonBackgroundExample(iterations):
    B = PolygonBilliards.regularPolygon(nSides=5, singularityLen=10)
   
    lines = B.singularity().simplify()
    points = lines.pointSpread(700)
    prev = []

    for i in range(iterations):
        prev.append(points)
        points = B(points).simplify()

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.8, 2.2)
    B.plot(showEdges=True, color="black")
    points.plot(size=5)

    if iterations > 0:
        PointSet.union(*prev).plot(size=1, color="red")
    
    plt.show()

def regularPolygonExample(nSides):
    B = PolygonBilliards.regularPolygon(
        nSides=nSides,
        singularityLen=25
    )

    result = usingLines(
        B,
        iterations=200,
        edgeMethod=PolygonBilliards.REFLECT_NONE,
        useSymmetry=False
    )

    result.plot()
    B.plot(color="black")
    plt.show()

def irregularPolygonExample():
    B = PolygonBilliards([
        (0, 2),
        (-2, 0.9),
        (-2, 0),
        (-1.5, -1),
        (1.5, -1),
        (2, 0)
    ], singularityLen=25)

    result = usingLines(
        B,
        iterations=200,
        edgeMethod=PolygonBilliards.REFLECT_NONE,
        useSymmetry=False
    )

    result.plot()
    B.plot(color="black")
    plt.show()

def method1(iterations, nSides):
    B = PolygonBilliards.regularPolygon(nSides=nSides)
    points = usingPoints(B, iterations, 3500)

    points.plot()
    B.plot(color="black")
    plt.show()

def lineSplitting():
    B = PolygonBilliards.regularPolygon(nSides=5, singularityLen=4)
   
    B.plot(color="black")
    B.singularity().plot(size=1)
    line = LineSet([[3, 1], [0, 3]])
    line1 = line.within(B.regions[0])
    line2 = line.within(B.regions[4])

    line1.plot(color="g", size=3)
    line2.plot(color="r", size=3)
    plt.show()

    B.plot(color="black")
    B(B.singularity()).plot(size=1)
    line1 = B(line1)
    line2 = B(line2)

    line1.plot(color="g", size=3)
    line2.plot(color="r", size=3)
    plt.show()

def lineSegmentsDemo():
    B = PolygonBilliards.regularPolygon(nSides=5, singularityLen=4)
   
    lines = B.singularity().simplify()

    allLines = []

    for i in range(10):
        B.plot(color="black")
        lines.union(*allLines).plot(size=1, pointSize=50, showPoints=True)
        plt.show()

        allLines.append(lines)
        lines = B(lines).simplify()


def linesVsPoints(iterations, nSides):
    B = PolygonBilliards.regularPolygon(nSides=nSides)
    points, pointCounts = usingPoints(B, iterations, 500*nSides, trackMemory=True)

    lines, lineCounts = usingLines(B, iterations, trackMemory=True)    

    plt.xlim(5.6, 6)
    plt.ylim(2.8, 3.2)    
    points.plot(size=10)
    plt.show()

    plt.xlim(5.6, 6)
    plt.ylim(2.8, 3.2)
    lines.plot(size=1)
    plt.show()

    plt.plot(pointCounts, label="Points")
    plt.plot(lineCounts, label="Lines")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Points/Lines")
    plt.legend()
    plt.show()
    
def redundantLines():
    B = PolygonBilliards.regularPolygon(nSides=5, singularityLen=4)
   
    plt.xlim(-0.2,2.3)
    plt.ylim(0.2,2.27)
    B.plot(color="black")
    lines = B.singularity()
    lines.plot(size=2, color="r")
    plt.show()

    for i in range(4):
        lines = B(lines)

        plt.xlim(-0.2,2.3)
        plt.ylim(0.2,2.27)
        B.plot(color="black")
        B.singularity().plot(size=1, color="black")
        lines.plot(color="r", size=2)
        plt.show()

def results(iterations, nSides):
    #Method 1 (baseline)
    B = PolygonBilliards.regularPolygon(
        nSides=nSides,
        singularityLen=25
    )
    result1, counts1 = usingPoints(
        B,
        iterations=iterations,
        seedPoints=100 * nSides,
        trackMemory=True
    )

    #Method 2 (line segments)
    B = PolygonBilliards.regularPolygon(
        nSides=nSides,
        singularityLen=25
    )
    result2, counts2 = usingLines(
        B,
        iterations=iterations,
        edgeMethod=PolygonBilliards.REFLECT_BOTH,
        useSymmetry=False,
        trackMemory=True
    )

    #Method 3 (remove redundant line segments)
    B = PolygonBilliards.regularPolygon(
        nSides=nSides,
        singularityLen=25
    )
    result3, counts3 = usingLines(
        B,
        iterations=iterations,
        edgeMethod=PolygonBilliards.REFLECT_NONE,
        useSymmetry=False,
        trackMemory=True
    )

    #Method 4 (symmetry)
    B = PolygonBilliards.regularPolygon(
        nSides=nSides,
        singularityLen=25
    )
    result4, counts4 = usingLines(
        B,
        iterations=iterations,
        edgeMethod=PolygonBilliards.REFLECT_NONE,
        useSymmetry=True,
        trackMemory=True
    )

    result1.plot()
    result2.plot()
    result3.plot(color="g")
    B.plot(color="black")
    plt.show()

    plt.plot(counts1, label="Method 1")
    plt.plot(counts2, label="Method 2")
    plt.plot(counts3, label="Method 3")
    plt.plot(counts4, label="Method 4")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # ovalBillardAnim()
    # pentagonSinglePointAnim()
    # piecewiseIsometryVisual()
    # dualValuedVisual()
    # pentagonBackgroundExample(0)
    # pentagonBackgroundExample(1)
    # pentagonBackgroundExample(10)
    # pentagonBackgroundExample(100)
    # regularPolygonExample(3)
    # regularPolygonExample(4)
    # regularPolygonExample(5)
    # regularPolygonExample(6)
    # regularPolygonExample(7)
    # regularPolygonExample(8)
    # irregularPolygonExample()
    # method1(100, 7)
    # method1(200, 7)
    lineSplitting()
    # lineSegmentsDemo()
    # linesVsPoints(100, 7)
    # redundantLines()
    # results(100, 5)
    # results(400)
    # results(1000)
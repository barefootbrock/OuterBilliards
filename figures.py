from numpy import sign
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
    pts = pts | B(pts)

    B.plot(showEdges=True, color="black")
    B.singularity().plot(size=1)
    pts.plot(size=100)
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

def irregularPolygonExample():
    B = PolygonBilliards([
        (0, 2),
        (-2, 0.9),
        (-2, 0),
        (-1.5, -1),
        (1.5, -1),
        (2, 0)
    ], singularityLen=25)

    lines = B.singularity().simplify()

    allLines = [lines]

    for i in range(200):
        lines = B(lines).simplify()
        allLines.append(lines)
        print("Iteration", i)

    B.plot(color="black")
    LineSet.union(*allLines).plot()
    plt.show()

def heptagonMethod1(iterations):
    start = time.time()
    B = PolygonBilliards.regularPolygon(singularityLen=25)
   
    lines = B.singularity().simplify()
    points = lines.pointSpread(700)
    prev = [points]
    
    for i in range(iterations):
        points = B(points).simplify()
        prev.append(points)
    
    result = PointSet.union(*prev, simplfy=False)
    
    print("Time:", time.time() - start)
    print("Memory:", sum(p.memory() for p in prev) / 1024**2)
    print("Memory:", result.memory() / 1024**2)

    result.plot()
    B.plot(color="black")
    plt.show()

    plt.plot([len(p) for p in prev])
    plt.show()

def lineSplitting():
    B = PolygonBilliards.regularPolygon(nSides=5, singularityLen=4)
   
    B.plot(color="black")
    B.singularity().plot(size=1)
    line = LineSet([[3, 1], [0, 3]])
    line.plot(color="r", size=3)
    plt.show()

    B.plot(color="black")
    B.singularity().plot(size=1)
    line = B(line)
    line.plot(color="r", size=3)
    plt.show()

def linesVsPoints(iterations):
    B = PolygonBilliards.regularPolygon()
   
    lines = B.singularity().simplify()
    points = lines.pointSpread(700)
    
    allPoints = []
    allLines = []
    pointMem = []
    lineMem = []

    for i in range(iterations):
        points = B(points).simplify()
        lines = B(lines).simplify()
        allPoints.append(points)
        allLines.append(lines)
        pointMem.append(len(points))
        lineMem.append(len(lines) * 2)
    
    points = PointSet.union(*allPoints)
    lines = LineSet.union(*allLines)

    plt.xlim(5.6, 6)
    plt.ylim(2.8, 3.2)    
    points.plot(size=10)
    plt.show()

    plt.xlim(5.6, 6)
    plt.ylim(2.8, 3.2)
    points.plot(size=10)
    lines.plot(size=1)
    plt.show()

    plt.plot(pointMem)
    plt.plot(lineMem)
    plt.show()
    
def redundantLines():
    B = PolygonBilliards.regularPolygon(nSides=5, singularityLen=4)
   
    plt.xlim(-0.2,2.3)
    plt.ylim(0.2,2.27)
    B.plot(color="black")
    lines = B.singularity()
    lines.plot(size=1, color="r")
    plt.show()

    for i in range(4):
        lines = B(lines)

        plt.xlim(-0.2,2.3)
        plt.ylim(0.2,2.27)
        B.plot(color="black")
        B.singularity().plot(size=1, color="black")
        lines.plot(color="r", size=2)
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
    # irregularPolygonExample()
    # heptagonMethod1(100)
    # heptagonMethod1(200)
    # lineSplitting()
    linesVsPoints(200)
    # redundantLines()
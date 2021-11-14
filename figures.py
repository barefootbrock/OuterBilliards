from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt

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
    point.plot(size=25)
    plt.pause(5)

    for i in range(6):
        point = B(point)
        pts.append(point[0,:])
        
        fig.clear()
        B.plot(showEdges=True, color="black")
        lines = LineSet.connect(pts)
        lines.plot(showPoints=True, pointSize=25)
        plt.xlim(-5, 5)
        plt.ylim(-3, 3)
        plt.pause(2)
    
    plt.show()

def pentagonSinglePointAnim():
    B = PolygonBillards.regularPolygon(nSides=5)

    fig = plt.figure()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # plt.axes("equal")

    point = PointSet([0.9, 1.2])
    pts = [point[0,:]]

    B.plot(showEdges=True, color="black")
    point.plot(size=25)
    plt.pause(5)

    for i in range(6):
        point = B(point)
        pts.append(point[0,:])
        
        fig.clear()
        B.plot(showEdges=True, color="black")
        lines = LineSet.connect(pts)
        lines.plot(showPoints=True, pointSize=25)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.pause(2)
    
    plt.show()

def heptagonMethod1():
    B = PolygonBillards.regularPolygon(singularityLen=5)
   
    lines = B.singularity().simplify()
    points = lines.pointSpread(700)
    
    for i in range(100):
        points = B(points).simplify()
    
    points.plot()
    plt.show()

if __name__ == "__main__":
    # ovalBillardAnim()
    # pentagonSinglePointAnim()
    heptagonMethod1()
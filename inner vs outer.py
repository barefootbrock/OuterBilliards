from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt

from geometry import LineSet
from innerbilliards import InnerBilliards
from outerbilliards import SmoothBilliards as OuterBilliards

fDist = 1.5
minor1 = 2
major1 = (minor1**2 + fDist**2)**0.5

Bi = InnerBilliards(lambda t: [
    major1 * cos(2*pi*t),
    minor1 * sin(2*pi*t)
])

#Make inner ellipse have same foci
minor2 = 1
major2 = (minor2**2 + fDist**2)**0.5
Bo = OuterBilliards(lambda t: [
    major2 * cos(2*pi*t),
    minor2 * sin(2*pi*t)
])

pointO = [major2, -(1 - major2**2/major1**2)**0.5 * minor1]
pointI, vel = pointO, [0, 1]

ptsI = [pointI]
ptsO = [pointO]

for i in range(10):
    pointI, vel = Bi(pointI, vel)
    pointO = Bo(pointO)
    
    ptsI.append(pointI[0,:])
    ptsO.append(pointO[0,:])


Bi.plot(showEdges=True, color="black", alpha=0.1, nPts=1000)
Bo.plot(showEdges=True, color="r", alpha=0.4, nPts=1000)

LineSet.connect(ptsI).plot(color="g", size=1, showPoints=True, pointSize=15)
LineSet.connect(ptsO).plot(color="b--", size=1, showPoints=True, pointSize=15)

plt.show()
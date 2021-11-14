from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt

iterations = 250
nSides = 7
singularityLen = 50

B = PolygonBillards.regularPolygon(
    nSides=nSides,
    singularityLen=singularityLen,
    edgeMethod='farthest'
)

lines = B.singularity().simplify()

allLines = [lines]

for i in range(iterations):
    lines = B(lines).simplify()
    allLines.append(lines)
    print("Iteration", i)

B.plot(color="black")
LineSet.union(allLines).plot()
plt.show()
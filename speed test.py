import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
import utils
from billiards import BilliardPoints, BilliardLines
import time

iterations = 100
cutLength = 25
addPoints = True
pointsPerCut = 100
addEvery = 1
removeEvery = 1


billiard = BilliardPoints(cutLength=cutLength)
iterator = billiard.iterator(
    iterations=iterations,
    seedPoints=pointsPerCut,
    addPoints=addPoints,
    addPointsEvery=addEvery,
    removeDuplicatesEvery=removeEvery
)
start = time.time()

for points in iterator:
    pass

pointTime = time.time() - start
print("Time for points (s):", pointTime)
print("Number of points:", len(points))
print("Memory used:", points.size * points.itemsize)


utils.plotPoints(points)
plt.title("Points %i iter\n%.1fs %.2fMB" % (
    iterations,
    pointTime, points.size * points.itemsize / 1024**2
))

plt.show()
print()

billiard = BilliardLines(cutLength=cutLength)
iterator = billiard.iterator(
    iterations=iterations,
    addPoints=addPoints,
    addPointsEvery=addEvery,
    removeDuplicatesEvery=removeEvery
)
start = time.time()

for lines in iterator:
    pass

lineTime = time.time() - start
print("Time for lines (s):", lineTime)
print("Number of lines:", len(lines))
print("Memory used:", lines.size * lines.itemsize)

utils.plotLines(lines)
plt.title("Lines %i iter\n%.1fs %.2fMB" % (
    iterations,
    lineTime, lines.size * lines.itemsize / 1024**2
))

plt.show()
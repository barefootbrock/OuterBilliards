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


billiardP = BilliardPoints(cutLength=cutLength)
iteratorP = billiardP.iterator(
    iterations=iterations,
    seedPoints=pointsPerCut,
    addPoints=addPoints,
    addPointsEvery=addEvery,
    removeDuplicatesEvery=removeEvery
)

billiardL = BilliardLines(cutLength=cutLength)
iteratorL = billiardL.iterator(
    iterations=iterations,
    addPoints=addPoints,
    addPointsEvery=addEvery,
    removeDuplicatesEvery=removeEvery
)

for i, (points, lines) in enumerate(zip(iteratorP, iteratorL)):
    print(i, len(points), len(lines))

utils.plotPoints(points)
utils.plotLines(lines)

# plt.savefig("Sep 20/lines vs points.png", dpi=600)

plt.show()
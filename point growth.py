import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
import utils
from billiards import BilliardPoints, BilliardLines
import time

billiard = BilliardLines(cutLength=25)
iterator = billiard.iterator(
    iterations=110,
    addPoints=False,
    removeDuplicates=False
)
pointCounts = []

for i, points in enumerate(iterator):
    pointCounts.append(len(points))
    print(i, len(points))

utils.plotLines(points)
plt.show()

plt.plot(pointCounts)
plt.show()

np.savetxt(
    "data6.csv",
    pointCounts,
    delimiter=",",
    fmt="%.5e")
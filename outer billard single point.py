import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
import utils
from billiards import BilliardPoints
from scipy.optimize import curve_fit

seed = np.array([[-1.6, -cos(pi/7)]])

billiard = BilliardPoints(cutLength=10)
iterator = billiard.iterator(seedPoints=seed, iterations=14,
                             addPoints=False, removeDuplicates=False)
fig = plt.figure()
pointHist = []

background = np.loadtxt(
    "data/lines 200 iter.csv",
    delimiter=",").reshape((-1, 2, 2))

for i, points in enumerate(iterator):
    print(i, points)
    fig.clear()
    plt.title("%i iterations, %i points" % (i, len(points)))
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    utils.plotLines(background, color="black", size=0.1)
    utils.plotPoints(points, color="red", size=25)
    plt.pause(0.5)

    pointHist.append(len(points))

    if i == 0:
        plt.pause(5)

plt.show()

x = np.arange(len(pointHist))
y = np.array(pointHist)
f = lambda x, n: 2**(x/n)
popt, pcov = curve_fit(f, x, y)
n = popt[0]

plt.title("Points vs. Iteration")
plt.plot(pointHist, label="# of points")
# plt.plot(x, f(x, n), label="y=2^(x/%.2f)" % n)
plt.legend()
plt.show()
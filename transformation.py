import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
from functools import reduce
import utils

class PiecewiseIsometry:
    def __init__(self, mats, regions):
        """Mats are matrices that define isometries"""
        self.mats = mats
        self.regions = regions
   
    def __call__(self, obj):
        results = []
        for mat, region in zip(self.mats, self.regions):
            results.append(obj.within(region).transform(mat))
        
        return np.concatenate(results).view(type(obj))
    
    def singularity(self):
        """Line segments on boundries of regions"""
        lines = [r.boundryEdges() for r in self.regions]
        lines = reduce(lambda a, b: a | b, lines)
        
        return lines



if __name__ == "__main__":
    from geometry import LineSet, Region
    import matplotlib.pyplot as plt

    A = np.array([0.5, 0.5*tan(2*pi/5)])
    B = np.array([0.0, 0.0])
    C = np.array([1.0, 0.0])
    D = np.array([cos(pi/5), sin(pi/5)])

    S0 = np.array([0.5, 0.5*tan(3*pi/10)])
    S1 = np.array([0.5, 0.5*tan(pi/10)])

    T0 = np.array([[+cos(4*pi/5), +sin(4*pi/5)],
                   [-sin(4*pi/5), +cos(4*pi/5)]])
    T0 = utils.matrixAboutPoint(T0, S0)

    T1 = np.array([[+cos(4*pi/5), -sin(4*pi/5)],
                   [+sin(4*pi/5), +cos(4*pi/5)]])
    T1 = utils.matrixAboutPoint(T1, S1)

    R0 = Region([A, B, D, A], S0, [True, False, True])
    
    R1 = Region([B, C, D, B], S1, [True, True, True])

    iso = PiecewiseIsometry((T0, T1), (R0, R1))

    plt.axis("equal")

    lines = iso.singularity().simplify()
    lines.plot()
    plt.show()

    allLines = [lines]

    for i in range(1000):
        lines = iso(lines)
        if i % 100 == 99:
            lines = lines.simplify()
            print(i, len(lines))
        allLines.append(lines)
    
    LineSet.union(allLines).plot()
    plt.show()
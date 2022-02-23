import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import utils
from geometry import PointSet, Region
from params import params

try:
    from scipy.optimize import minimize_scalar
except ImportError:
    print("Scipy not found, SmoothBilliards will not work")

class InnerBilliards:
    def __init__(self, r):
        """
        Smooth paramitized curve(<f(t), g(t)>)
        """
        self.rFunc = r

        self.center = np.mean(self.r(np.linspace(0, 1)), 0)
    
    def r(self, t):
        if np.isscalar(t):
            return np.asarray(self.rFunc(t), dtype=params.dtype).reshape((2,))
        else:
            return np.asarray(self.rFunc(t), dtype=params.dtype).reshape((2, -1)).T
    
    def __call__(self, points, directions):
        hit, newDir = self.bounce(points, directions)
        return hit, newDir
    
    def bounce(self, points, directions):
        """
        Point on curve whose tangent line passes though point
        1: gradient assent to maximize (r(t) - point) . v(t)
        Use Newton's method 5-10 iterations:
            x = x - f(x)/f'(x)
            t = t - ((r(t) - point) X v(t)) ./ ((r(t) - point) X a(t))
        """

        def err(t, point, direction):
            #Error function to minimize
            return -np.dot(
                utils.normalize(self.r(t) - [point]),
                direction
            )
        
        tVals = []
        dirs = []
        for point, direction in zip(PointSet(points), PointSet(directions)):
            t = float(minimize_scalar(err, args=(point, direction), tol=1e-12).x % 1)
            tangent = (self.r(t + params.eps) - self.r(t)) / params.eps
            newDir = 2 * np.dot(direction, tangent) / np.dot(tangent, tangent) * tangent - direction
            tVals.append(t)
            dirs.append(newDir)
        
        t = np.asarray(tVals, dtype=params.dtype)

        return self.r(t), np.array(dirs)

    def plot(self, nPts=100, **kwargs):
        t = np.linspace(0, 1, nPts)
        pts = self.r(t)
        Region(pts, self.center).plot(**kwargs)


if __name__ == "__main__":
    from geometry import LineSet
    import matplotlib.pyplot as plt

    B = InnerBilliards(lambda t: [
        5*cos(2*pi*t),
        4*sin(2*pi*t)
    ])

    point, vel = [4, -2.4], [0, 1]
    pts = [point]
    fig = plt.figure()
    plt.xlim(-6, 6)
    plt.ylim(-5, 5)
    plt.pause(5)

    for i in range(9):
        point, vel = B(point, vel)
        print(point)
        pts.append(point[0,:])

        # fig.clear()
        # B.plot(showEdges=True, color="black")
        # lines = LineSet.connect(pts)
        # lines.plot(showPoints=True, pointSize=15)
        # plt.pause(0.1)
    
    # plt.show()


    pts = LineSet.connect(pts).pointSpread(200)
    pts.plot(size=15)

    for i, (x, y) in enumerate(pts):
        fig.clear()
        plt.xlim(-6, 6)
        plt.ylim(-5, 5)
        B.plot(showEdges=True, color="black", alpha=0.1)
        plt.scatter([x], [y], s=25, c='r')

        lines = LineSet.connect(pts[:i+1,...])
        lines.plot(size=2)

        plt.pause(0.01)
    
    plt.show()
from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt
from algorithms import usingLines

B = PolygonBillards.regularPolygon(nSides=7)

lines = usingLines(
    1000,
    sides=7,
    splitMode='farthest',
    useSymmetry=True
)

lines.plot()
B.plot(color="black")
plt.show()
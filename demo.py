from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt
from algorithms import usingLines

B = PolygonBilliards.regularPolygon(nSides=7)

lines = usingLines(
    1000,
    sides=7,
    splitMode='remove',
    useSymmetry=True
)

lines.plot()
B.plot(color="black")
plt.show()
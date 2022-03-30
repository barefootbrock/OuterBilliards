from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt
from algorithms import usingLines, usingPoints

B = PolygonBilliards.regularPolygon(
    nSides=7,
    singularityLen=25
)

usingLines(
    B,
    iterations=100
).plot()

usingPoints(
    B,
    iterations=100,
    seedPoints=1400
).plot(size=5)

B.plot(color="black")
plt.show()
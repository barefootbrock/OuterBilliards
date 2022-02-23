from outerbilliards import *
from geometry import *
import matplotlib.pyplot as plt
from algorithms import usingLines, usingPoints

B = PolygonBilliards.regularPolygon(
    nSides=7,
    singularityLen=25
)

result, counts = usingLines(
    B,
    iterations=500,
    edgeMethod=PolygonBilliards.REFLECT_NONE, #Can be REFLECT_BOTH, REFLECT_FAR, REFLECT_NONE
    useSymmetry=True,
    trackMemory=True
)

#Uncomment the following to use points instead
# result, counts = usingPoints(
#     B,
#     iterations=200,
#     seedPoints=700,
#     trackMemory=True
# )


result.plot()
B.plot(color="black")
plt.show()

plt.plot(counts)
plt.show()
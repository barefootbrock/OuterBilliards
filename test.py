import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
import utils
from billiards import BilliardPoints, BilliardLines
import time

billiard = BilliardLines(cutLength=250)
iterator = billiard.iterator(
    iterations=100,
    addPoints=False,
    removeDuplicates=True
)
next(iterator)

t = time.time()
prevLines = []
for i, lines in enumerate(iterator):
    print(i, len(lines))
    prevLines.append(lines)

allLines = np.concatenate(prevLines)
allLines = utils.mergeOverlapingSegments(allLines)
print(time.time() - t)

utils.plotLines(allLines)
plt.show()







# billiard = BilliardLines(cutLength=25)
# iterator1 = billiard.iterator(
#     iterations=50,
#     addPoints=True,
#     removeDuplicates=True
# )

# iterator2 = billiard.iterator(
#     iterations=50,
#     addPoints=False,
#     removeDuplicates=True
# )

# prevLines = []

# next(iterator1)
# next(iterator2)

# for lines1, lines2 in zip(iterator1, iterator2):
#     print(len(lines1), len(lines2))
#     prevLines.append(lines2)

# lines1 = utils.mergeOverlapingSegments(lines1)
# print(len(lines1))

# lines2 = np.concatenate(prevLines)
# print(len(lines2))
# lines2 = utils.mergeOverlapingSegments(lines2)
# print(len(lines2))

# utils.plotLines(lines1, color="r")
# utils.plotLines(lines2, color="b")

# plt.show()

# print(np.allclose(lines1, lines2))
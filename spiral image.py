import matplotlib.pyplot as plt
import utils
from billiards import BilliardPoints, BilliardLines

billiard = BilliardLines(cutLength=85)
fig = plt.figure()

for i, lines in enumerate(billiard.iterator(iterations=200, addPoints=False)):
    fig.clear()
    plt.title("%i iterations, %i lines" % (i, len(lines)))
    plt.xlim(-90, 90)
    plt.ylim(-90, 90)
    utils.plotLines(lines, color="black")
    plt.pause(0.1)

    if i == 0:
        plt.pause(10)

plt.show()
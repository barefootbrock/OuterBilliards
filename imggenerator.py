import numpy as np
from numpy import linalg
from numpy import sin, cos, tan, pi, inf
import matplotlib.pyplot as plt
import utils
from billiards import BilliardPoints, BilliardLines
import time

def run(iterator, verbose=True, keepAll=True):
    start = time.time()
    hist = []
    sizes = []
    for i, out in enumerate(iterator):
        if keepAll:
            hist.append(out)
        sizes.append(out.shape[0])
        if verbose:
            print(i, out.shape)
        # if i % 100 == 0:
        #     print(time.time() - start, sum(sizes))
    
    if keepAll:
        out = np.concatenate(hist, 0)
        # out = np.concatenate(hist[-5:], 0)

    if len(out.shape) == 2: #points
        out = utils.removeDuplicatePts(out)
    elif len(out.shape) == 3: #lines
        out = utils.mergeOverlapingSegments(out)
    
    print("That took %.3fs" % (time.time() - start))
    
    if len(out.shape) == 2:
        utils.plotPoints(out)
    elif len(out.shape) == 3:
        utils.plotLines(out)
    
    plt.show()

    plt.plot(sizes)
    plt.show()

    return out

if __name__ == "__main__":
    run(BilliardLines().iterator(addPoints=False))
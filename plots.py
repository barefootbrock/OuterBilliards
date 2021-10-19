import numpy as np
import matplotlib.pyplot as plt

y1 = np.loadtxt(
    "Oct 5/data4.csv",
    delimiter=",")
    
y2 = np.loadtxt(
    "Oct 5/data5.csv",
    delimiter=",")
    
y3 = np.loadtxt(
    "Oct 5/data6.csv",
    delimiter=",")

plt.xlabel("Iteration")
plt.ylabel("log(line count)")
plt.plot(np.log10(y1), label="addPoints=no, removeDuplicates=yes")
plt.plot(np.log10(y2), label="addPoints=yes, removeDuplicates=yes")
plt.plot(np.log10(y3), label="addPoints=no, removeDuplicates=no")
plt.legend()
plt.show()
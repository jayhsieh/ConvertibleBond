from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
fig = plt.figure()
ax = fig.add_subplot(111)

file_path = "D:/Data/timecost_debug.txt"
data = np.loadtxt(file_path)
plt.plot(data[:, 0], data[:, 1], 'o-')

file_path = "D:/Data/timecost_release.txt"
data = np.loadtxt(file_path)
plt.plot(data[:, 0], data[:, 1], 's-')

ratio = np.divide(data[:,0], data[0,0])
sqrratio = np.square(ratio)
plt.plot(data[:, 0], np.multiply(data[0, 1], ratio), 'r--')
plt.plot(data[:, 0], np.multiply(data[0, 1], sqrratio), '>-')
plt.legend(["Time cost of debugging version", "Time cost of release version",
            "linear time cost", "quadratic time cost"], loc = "upper left")
#ax.set_yscale('log')
plt.ylabel("Time cost [ms]")
plt.xlabel("Number of grid points")
plt.show()
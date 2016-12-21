from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.set_xscale('log')

#fnames = ['D:/Data/value_cn_0.005dx.txt', 'D:/Data/value_cn_0.01dx.txt', 'D:/Data/value_cn_0.05dx.txt', 'D:/Data/value_cn_0.1dx.txt']
fnames = ['D:/value.txt']
color = ['r', 'g', 'b', 'k']
for i in range(0, len(fnames)):
    print(fnames[i])
    file_path = fnames[i]
    data = np.loadtxt(file_path)
    plt.plot(data[0:-2, 0], np.subtract(data[0:-2, 1], data[-1,1]),'o-')
    err = np.subtract(data[0:-2, 1], data[-1,1])
    for i in range(1, len(err)):
        slope = np.log2(err[i-1]/err[i])
        print(slope)
    print(np.subtract(data[0:-2, 1], data[-1,1]))
#plt.legend(["cn 0.005", 'cn 0.01', 'cn 0.05', 'cn 0.1'], loc = "lower left")
plt.legend(["Numerical Error"], loc = "upper right")
plt.ylabel("Error")
plt.xlabel("Number of grid points")
plt.show()
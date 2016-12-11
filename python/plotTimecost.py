from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
fig = plt.figure()
ax = fig.add_subplot(111)

file_path = "D:/timecost.txt"
data = np.loadtxt(file_path)
plt.plot(data[:, 0], data[:, 1], 'bo-')

p = np.polyfit(data[:,0], data[:,1], 2)
print(p)
v2 = np.multiply(data[:,0], data[:,0])
v2 = np.multiply(v2, p[0])
v1 = np.multiply(data[:,0], p[1])
fit = np.add(v2, v1)
fit = np.add(fit, p[2])
plt.plot(data[:,0], fit, 'b--')


file_path = "D:/Data/timecost_release.txt"
data = np.loadtxt(file_path)
plt.plot(data[:, 0], data[:, 1], 'rs-')
p = np.polyfit(data[:,0], data[:,1], 2)
v2 = np.multiply(data[:,0], data[:,0])
v2 = np.multiply(v2, p[0])
v1 = np.multiply(data[:,0], p[1])
fit = np.add(v2, v1)
fit = np.add(fit, p[2])
plt.plot(data[:,0], fit, 'r--')

plt.legend(["Time cost of debug version", "Fit curve for debug version",
            "Time cost of release version", "Fit curve for release version"], loc = "upper left")
plt.ylabel("Time cost [ms]")
plt.xlabel("Number of grid points")
plt.show()
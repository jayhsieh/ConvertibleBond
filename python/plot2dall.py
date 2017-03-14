from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

path = "D:/Results/"
fnames = "sol-0"
legends = [fnames]
fig = plt.figure()

V = np.loadtxt(path + fnames + ".txt")
#V = np.matrix.transpose(V)
msize = V.shape
Smax = 100
kappa = 1
Nt = msize[0]
Nx = msize[1]
S = np.linspace(0, Smax, Nx)

print(msize, Nx, Nt)

for i in range(0, Nt, 2):
    plt.plot(S, V[i, :], color = (0, i/Nt, 0, 1))

plt.legend(legends, loc = "upper left")
plt.xlabel("Stock Price")
plt.ylabel("Convertible Bond Value")
plt.set_cmap('hot')
#plt.plot(S, kappa*S, 'r--')
axes = plt.gca()
#axes.set_xlim([20, 180])
#axes.set_ylim([90, 180])
plt.show()
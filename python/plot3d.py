from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

#CBprice = np.loadtxt('D:/ConvertibleBond/cs/ConvertibleBond/ConvertibleBond/bin/Debug/CBsol.txt')
V = np.loadtxt('D:/CBsol.txt')
V = np.matrix.transpose(V)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
msize = V.shape
T = 5
B = 100
Smax = 5*B
Nt = msize[1]
Nx = msize[0]
t = np.linspace(T, 0, Nt)
S = np.linspace(0, Smax, Nx)
X, Y = np.meshgrid(t, S)
Vmin = 90
Vmax = 160
V[V < Vmin] = NaN
V[V > Vmax] = NaN
#V[Y > 160] = NaN;
ax.set_zlim(Vmin, Vmax)
ax.set_ylim(20, 160)
surf = ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=cm.coolwarm,
                   linewidth=0, antialiased=False, vmin = Vmin, vmax = Vmax)
ax.set_xlabel("Time [years]")
ax.set_ylabel("Stock price [$]")
ax.set_zlabel("Convertible Bond Price [$]")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
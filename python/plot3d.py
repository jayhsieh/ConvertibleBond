from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

CBprice = np.loadtxt('D:/ConvertibleBond/cs/ConvertibleBond/ConvertibleBond/bin/Debug/COCBsol.txt')
#CBprice = np.loadtxt('D:/CBsol.txt')
CBprice = np.matrix.transpose(CBprice)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
msize = CBprice.shape
T = 5
B = 101
Smax = 160
Nt = msize[1]
Nx = msize[0]
t = np.linspace(T, 0, Nt)
S = np.linspace(0, Smax, Nx)
X, Y = np.meshgrid(t, S)
surf = ax.plot_surface(X, Y, CBprice, rstride=1, cstride=1, cmap=cm.coolwarm,
                   linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

V = np.loadtxt('D:/Results/COCBsol.txt')
t = np.loadtxt('D:/Results/time.txt')

#parameters for CB
# Vmin = 95; Vmax = 165
# Ymin = 20; Ymax = 160
# Smax = V[0,-1]
# angle = -30
# meshstride = 8

#parameters for COCB
Vmin = 0; Vmax = amax(V)
Ymin = 0; Ymax = 200
print(amax(V))
Smax = 500
angle = 20
meshstride = 20

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
msize = V.shape
T = t[0]
print(T, msize, len(t))

Nt = msize[0]
Nx = msize[1]
S = np.linspace(0, Smax, Nx)
X, Y = np.meshgrid(S, t)

V[V < Vmin] = NaN
V[V > Vmax] = NaN
V[X < Ymin] = NaN
V[X > Ymax] = NaN
ax.set_zlim(Vmin, Vmax)
ax.set_ylim(Ymin, Ymax)

surf = ax.plot_surface(Y, X, V, cmap=cm.coolwarm, cstride=round(Nx/80), rstride=1, alpha=0.99,
                   linewidth=0.1, antialiased=False, vmin = Vmin, vmax = Vmax)
#ax.plot_wireframe(Y, X, V, rstride=meshstride, cstride=meshstride)
ax.set_xlabel("Time [years]")
ax.set_ylabel("Stock price [$]")
ax.set_zlabel("CB [$]")
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(30, angle)
plt.show()
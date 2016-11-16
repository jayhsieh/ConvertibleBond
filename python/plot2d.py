from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

path = "D:/"
#fnames = ["CBsol_zero", "CBsol_coupon", "CBsol_put", "CBsol_callput"]
#decros = ["-", ".", ".-", "--"];
#legends = ["Zero-coupon", "Coupon payment", "Coupon plus put", "Coupon plus put and call"];
fnames = ["CBsol"]
decros = ["-"]
legends = ["CB"]
fig = plt.figure()

for i in range(0, len(fnames)):
    V = np.loadtxt(path + fnames[i] + ".txt")
    V = np.matrix.transpose(V)
    msize = V.shape
    B = 100
    Smax = 500
    Nx = msize[0]
    Nt = msize[1]
    S = np.linspace(0, Smax, Nx)
    plt.plot(S, V[:, -1], decros[i])

plt.legend(legends, loc = "upper left")
plt.xlabel("Stock Price")
plt.ylabel("Convertible Bond Value")
#axes = plt.gca()
#axes.set_xlim([20, 160])
#axes.set_ylim([50, 160])
plt.show()
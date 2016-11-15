from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

CBprice = np.loadtxt('D:/ConvertibleBond/cs/ConvertibleBond/ConvertibleBond/bin/Debug/COCBsol.txt')
CBprice = np.matrix.transpose(CBprice)

fig = plt.figure()
msize = CBprice.shape
B = 101
Smax = 500
Nx = msize[0]
Nt = msize[1]
S = np.linspace(0, Smax, Nx)


for k in range(1, Nt, 100):
    plt.plot(S, CBprice[:, k]);

plt.show()
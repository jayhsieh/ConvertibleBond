import numpy as np
import matplotlib.pyplot as plt

Smax = 500
a = np.linspace(Smax/99, Smax/99, 1)

for ai in a:
    #S = np.linspace(0, Smax, 200)
    x = np.linspace(0, 0.999, 200)
    S = np.divide(ai*x, 1 - x)
    plt.plot(x, S, 'o-')
plt.xlabel("S")
plt.ylabel("x")
plt.show()
import numpy as np

Lx = Ly = 20


def phi(x: float, y: float) -> float:
    x0 = 0.13 * Lx  # x center of the first wedget
    xs = 0.24 * Lx  # spacing between wedget
    R = 0.06 * Lx  # radius of the wedget
    xc = x0
    for n in range(4):
        xc = x0 + n * xs
        if abs(x - xc) <= xs / 2:
            break
    return (x - xc) ** 2 + y * y - R * R


x = np.linspace(0, Lx, 50)
y = np.linspace(0, Ly, 100)
Y, X = np.meshgrid(y, x)
PHI = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        PHI[i, j] = phi(X[i, j], Y[i, j])

import matplotlib.pyplot as plt

plt.contourf(X, Y, PHI, levels=20)
plt.colorbar(label="Level set value")
plt.show()

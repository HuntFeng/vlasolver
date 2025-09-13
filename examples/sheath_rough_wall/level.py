import numpy as np

R = 0.06
L = 1.0
x0 = 0.13
xs = 0.24

xc = np.arange(x0, L, xs)


def phi(x: float, y: float) -> float:
    # n = np.argmin(np.abs(x - xc))
    # return (x - xc[n]) ** 2 + y**2 - R**2
    x0 = 0.13
    # x center of the first wedget
    xs = 0.24
    # spacing between wedget
    R = 0.06
    # radius of the wedget
    xc = x0
    for n in range(4):
        xc = x0 + n * xs
        if abs(x - xc) <= xs / 2:
            break

    # minimum |x-xc|
    # min = 100.0
    # for n in range(4):
    #     if abs(x - (xc + xs)) < min:
    #         min = abs(x - xc)
    #         xc += xs
    #     else:
    #         break
    return (x - xc) ** 2 + y * y - R * R


x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
Y, X = np.meshgrid(y, x)
PHI = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        PHI[i, j] = phi(X[i, j], Y[i, j])

import matplotlib.pyplot as plt

plt.contourf(X, Y, PHI, levels=20)
plt.colorbar(label="Level set value")
plt.show()

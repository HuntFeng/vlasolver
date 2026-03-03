import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))

n = 64
nx, ny = 2 * n, n
Lx, Ly = 1, 0.5
dx = Lx / nx
dy = Ly / ny
G = 3

fig, ax = plt.subplots(2, 1, sharex=True)
for i in range(0, 401, 40):
    with h5py.File(
        f"{file_path}/../../data/advection/output_{n}_{i:03d}.h5",
        "r",
    ) as f:
        ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
    y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax[0].plot(x, ni[:, G], "o-", label=f"$y=0$, t={i}")
    ax[1].plot(x, ni[:, (ny // 2)], "o-", label=f"$y=0.25$, t={i}")
ax[1].set_xlabel("$x/L_x$")
ax[0].set_ylabel("$n$")
ax[0].set_ylabel("$n$")
ax[0].legend()
ax[1].legend()
plt.show()

for i in range(0, 401, 40):
    with h5py.File(
        f"{file_path}/../../data/advection/output_{n}_{i:03d}.h5",
        "r",
    ) as f:
        ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
    y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
    X, Y = np.meshgrid(x, y, indexing="ij")
    plt.figure()
    plt.pcolormesh(X, Y, ni, cmap="jet")
    plt.colorbar()
    plt.title(f"step = {i}")
    plt.xlabel("x")
    plt.ylabel("y")
plt.show()

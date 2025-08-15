import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 14,  # Base font size
        "axes.labelsize": 16,  # Size for x and y labels
        "axes.titlesize": 16,  # Size for plot titles
        "xtick.labelsize": 14,  # Size for x-axis tick labels
        "ytick.labelsize": 14,  # Size for y-axis tick labels
        "legend.fontsize": 14,  # Size for legend text
        "figure.titlesize": 16,  # Size for figure titles
    }
)

nx, ny, nvx, nvy = 10, 125, 30, 110
Lx, Ly, Lvx, Lvy = 20, 20, 8, 9
x_min, y_min, vx_min, vy_min = 0, 0, -4, -8
G = 3
step = 0
with h5py.File(
    f"{os.path.dirname(os.path.realpath(__file__))}/../../data/plasma_sheath/output_{step:03d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    fi = f["VTKHDF/CellData/fi"][:].reshape(
        nx + 2 * G, ny + 2 * G, nvx + 2 * G, nvy + 2 * G
    )

ni = ni[G:-G, G:-G]
phi = phi[G:-G, G:-G]
fi = fi[G:-G, G:-G, G:-G, G:-G]

dx, dy, dvx, dvy = Lx / nx, Ly / ny, Lvx / nvx, Lvy / nvy
x = np.arange(x_min + dx / 2, x_min + Lx, dx)
y = np.arange(y_min + dy / 2, y_min + Ly, dy)
vx = np.arange(vx_min + dvx / 2, vx_min + Lvx, dvx)
vy = np.arange(vy_min + dvy / 2, vy_min + Lvy, dvy)

VY, Y = np.meshgrid(vy, y)
plt.figure()
plt.contourf(Y, VY, fi[0, :, 0, :], cmap="jet", levels=50)
plt.colorbar(label="$f_i$")
plt.xlabel("$y$")
plt.ylabel("$v_y$")


plt.figure()
plt.plot(y, phi[0, :])
plt.xlabel("$y$")
plt.ylabel("$\\phi$")

plt.figure()
plt.plot(y, ni[0, :])
plt.xlabel("$y$")
plt.ylabel("$n_i$")

plt.show()

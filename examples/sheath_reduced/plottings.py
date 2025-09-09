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
Lx, Ly, Lvx, Lvy = 1, 1, 8, 9
x_min, y_min, vx_min, vy_min = 0, 0, -4, -8
G = 3
step = 5000
with h5py.File(
    f"{os.path.dirname(os.path.realpath(__file__))}/../../data/sheath_reduced/output_{step:04d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    fi = f["VTKHDF/CellData/fi"][:].reshape(
        nx + 2 * G, ny + 2 * G, nvx + 2 * G, nvy + 2 * G
    )

is_include_ghost = True
if is_include_ghost:
    dx, dy, dvx, dvy = Lx / nx, Ly / ny, Lvx / nvx, Lvy / nvy
    x = np.arange(x_min - G * dx + dx / 2, x_min + Lx + G * dx, dx)
    y = np.arange(y_min - G * dy + dy / 2, y_min + Ly + G * dy, dy)
    vx = np.arange(vx_min - G * dvx + dvx / 2, vx_min + Lvx + G * dvx, dvx)
    vy = np.arange(vy_min - G * dvy + dvy / 2, vy_min + Lvy + G * dvy, dvy)
else:
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
plt.contourf(Y, VY, fi[fi.shape[0] // 2, :, fi.shape[2] // 2, :], cmap="jet", levels=50)
plt.colorbar(label="$f_i$")
plt.xlabel("$y$")
plt.ylabel("$v_y$")

plt.figure()
plt.plot(y, phi[phi.shape[0] // 2, :])
plt.xlabel("$y$")
plt.ylabel("$\\phi$")

plt.figure()
plt.plot(y, ni[ni.shape[0] // 2, :], label="$n_i$")
plt.plot(y, np.exp(phi[ni.shape[0] // 2, :]), label="$n_e$")
plt.legend()
plt.xlabel("$y$")
plt.ylabel("$n$")

plt.show()

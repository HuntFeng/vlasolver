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
x_min, y_min = 0, 0
Lx, Ly = 1, 20.0

vx_min_e, vy_min_e = -4, -5
Lvx_e, Lvy_e = 8, 10
# in calculation, the ranges are devided by vr
vx_min_i, vy_min_i = -4, -8
Lvx_i, Lvy_i = 8, 9
G = 3
step = 0
with h5py.File(
    f"{os.path.dirname(os.path.realpath(__file__))}/../../data/sheath_full/output_{step:04d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    fi = f["VTKHDF/CellData/fi"][:].reshape(
        nx + 2 * G, ny + 2 * G, nvx + 2 * G, nvy + 2 * G
    )
    fe = f["VTKHDF/CellData/fe"][:].reshape(
        nx + 2 * G, ny + 2 * G, nvx + 2 * G, nvy + 2 * G
    )

is_include_ghost = True
if is_include_ghost:
    dx, dy = Lx / nx, Ly / ny
    x = np.arange(x_min - G * dx + dx / 2, x_min + Lx + G * dx, dx)
    y = np.arange(y_min - G * dy + dy / 2, y_min + Ly + G * dy, dy)
    dvx_e, dvy_e = Lvx_e / nvx, Lvy_e / nvy
    vx_e = np.arange(
        vx_min_e - G * dvx_e + dvx_e / 2, vx_min_e + Lvx_e + G * dvx_e, dvx_e
    )
    vy_e = np.arange(
        vy_min_e - G * dvy_e + dvy_e / 2, vy_min_e + Lvy_e + G * dvy_e, dvy_e
    )
    dvx_i, dvy_i = Lvx_i / nvx, Lvy_i / nvy
    vx_i = np.arange(
        vx_min_i - G * dvx_i + dvx_i / 2, vx_min_i + Lvx_i + G * dvx_i, dvx_i
    )
    vy_i = np.arange(
        vy_min_i - G * dvy_i + dvy_i / 2, vy_min_i + Lvy_i + G * dvy_i, dvy_i
    )
else:
    ni = ni[G:-G, G:-G]
    ne = ne[G:-G, G:-G]
    phi = phi[G:-G, G:-G]
    fi = fi[G:-G, G:-G, G:-G, G:-G]
    fe = fe[G:-G, G:-G, G:-G, G:-G]

    dx, dy = Lx / nx, Ly / ny
    x = np.arange(x_min + dx / 2, x_min + Lx, dx)
    y = np.arange(y_min + dy / 2, y_min + Ly, dy)

    dvx_e, dvy_e = Lvx_e / nvx, Lvy_e / nvy
    vx_e = np.arange(vx_min_e + dvx_e / 2, vx_min_e + Lvx_e, dvx_e)
    vy_e = np.arange(vy_min_e + dvy_e / 2, vy_min_e + Lvy_e, dvy_e)
    dvx_i, dvy_i = Lvx_i / nvx, Lvy_i / nvy
    vx_i = np.arange(vx_min_i + dvx_i / 2, vx_min_i + Lvx_i, dvx_i)
    vy_i = np.arange(vy_min_i + dvy_i / 2, vy_min_i + Lvy_i, dvy_i)

VY_e, Y = np.meshgrid(vy_e, y)
VY_i, Y = np.meshgrid(vy_i, y)
fig, ax = plt.subplots(1, 2)
ax[0].contourf(
    Y,
    VY_e,
    fe[fe.shape[0] // 2, :, fe.shape[2] // 2, :],
    cmap="jet",
    levels=50,
    vmin=0,
)
ax[0].set_xlabel("$y$")
ax[0].set_ylabel("$v_y$")
ax[0].set_title("$f_e$")
ax[1].contourf(
    Y,
    VY_i,
    fi[fi.shape[0] // 2, :, fi.shape[2] // 2, :],
    cmap="jet",
    levels=50,
    vmin=0,
)
ax[1].set_xlabel("$y$")
ax[1].set_ylabel("$v_y$")
ax[1].set_title("$f_i$")
fig.tight_layout()

plt.figure()
plt.plot(y, phi[phi.shape[0] // 2, :])
plt.xlabel("$y$")
plt.ylabel("$\\phi$")

plt.figure()
plt.plot(y, ne[ne.shape[0] // 2, :], label="$n_e$")
plt.plot(y, ni[ni.shape[0] // 2, :], label="$n_i$")
plt.legend()
plt.xlabel("$y$")
plt.ylabel("$n$")

plt.show()

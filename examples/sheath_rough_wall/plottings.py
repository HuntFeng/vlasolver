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

nx, ny = 50, 100
nvx, nvy = 50, 50
x_min, y_min = 0, 0
Lx, Ly = 20, 20
G = 3
step = 40000
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/sheath_rough_wall/output_{step:05d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    # fi = f["VTKHDF/CellData/fi"][:].reshape(
    #     nx + 2 * G, ny + 2 * G, nvx + 2 * G, nvy + 2 * G
    # )

is_include_ghost = True
if is_include_ghost:
    dx, dy = Lx / nx, Ly / ny
    x = np.arange(x_min - G * dx + dx / 2, x_min + Lx + G * dx, dx)
    y = np.arange(y_min - G * dy + dy / 2, y_min + Ly + G * dy, dy)
else:
    ni = ni[G:-G, G:-G]
    ne = ne[G:-G, G:-G]
    phi = phi[G:-G, G:-G]

    dx, dy = Lx / nx, Ly / ny
    x = np.arange(x_min + dx / 2, x_min + Lx, dx)
    y = np.arange(y_min + dy / 2, y_min + Ly, dy)

Y, X = np.meshgrid(y, x)
plt.figure()
plt.subplot(1, 2, 1)
plt.contourf(
    X,
    Y,
    ne,
    cmap="jet",
    levels=50,
    vmin=0,
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$n_e$")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.contourf(
    X,
    Y,
    ni,
    cmap="jet",
    levels=50,
    vmin=0,
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$n_i$")
plt.tight_layout()
plt.colorbar()
plt.savefig(f"{file_path}/number_density.png")

plt.figure()
plt.contourf(X, Y, phi, levels=15, cmap="jet")
plt.colorbar()
plt.contour(X, Y, phi, levels=15, colors="black", linestyles="solid")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$\\phi$")
plt.savefig(f"{file_path}/potential.png")

plt.show()

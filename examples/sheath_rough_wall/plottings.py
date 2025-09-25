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
x_min, y_min = 0, 0
Lx, Ly = 1, 1
G = 3
step = 8000
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/sheath_rough_wall/output_{step:04d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)

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
fig, ax = plt.subplots(1, 2)
ax[0].contourf(
    X,
    Y,
    ne,
    cmap="jet",
    levels=50,
    vmin=0,
)
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$y$")
ax[0].set_title("$n_e$")
ax[1].contourf(
    X,
    Y,
    ni,
    cmap="jet",
    levels=50,
    vmin=0,
)
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$y$")
ax[1].set_title("$n_i$")
fig.tight_layout()
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

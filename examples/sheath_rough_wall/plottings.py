import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

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
Lx, Ly = 1.0, 1.0  # they are 20 in simulation, but normalize them again in plots
G = 3
Ti = 0.1  # eV
Te = 1.0  # eV
Tr = Ti / Te

step = 40000
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/sheath_rough_wall/output_{step:05d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G) / (2 * Tr)

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

X, Y = np.meshgrid(x, y, indexing="ij")
plt.figure()
plt.contourf(
    X,
    Y,
    ne,
    cmap="jet",
    levels=20,
    vmin=0,
)
plt.colorbar()
plt.contour(X, Y, ne, levels=20, colors="black", linestyles="solid")
for xc in np.arange(0.13 * Lx, Lx, 0.24 * Lx, dtype=float):
    circle = Wedge(
        center=(xc, 0),
        r=0.06 * Lx,
        theta1=0,
        theta2=180,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
        zorder=10,  # ensure the wedge is on top of the contour
    )
    plt.gca().add_patch(circle)
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$n_e$")
plt.savefig(f"{file_path}/number_density_electron.png")

plt.figure()
plt.contourf(
    X,
    Y,
    ni - ne,
    cmap="jet",
    levels=20,
    vmin=0,
)
plt.colorbar()
plt.contour(X, Y, ni - ne, levels=20, colors="black", linestyles="solid")
for xc in np.arange(0.13 * Lx, Lx, 0.24 * Lx, dtype=float):
    circle = Wedge(
        center=(xc, 0),
        r=0.06 * Lx,
        theta1=0,
        theta2=180,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
        zorder=10,  # ensure the wedge is on top of the contour
    )
    plt.gca().add_patch(circle)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$\\rho$")
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.savefig(f"{file_path}/charge_density.png")

plt.figure()
plt.contourf(X, Y, phi, levels=20, cmap="jet")
plt.colorbar()
plt.contour(X, Y, phi, levels=20, colors="black", linestyles="solid")
for xc in np.arange(0.13 * Lx, Lx, 0.24 * Lx, dtype=float):
    circle = Wedge(
        center=(xc, 0),
        r=0.06 * Lx,
        theta1=0,
        theta2=180,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
        zorder=10,  # ensure the wedge is on top of the contour
    )
    plt.gca().add_patch(circle)
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$e\\phi/2k_BT_i$")
plt.savefig(f"{file_path}/potential.png")

plt.show()

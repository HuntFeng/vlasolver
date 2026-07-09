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

nx, ny = 128, 128
x_min, y_min = -1, -1
Lx, Ly = 2, 2
G = 3
is_include_circle = True

step = 1000
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/sheath_cylinder/output_{step:04d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)

dx = Lx / nx
dy = Ly / ny
x = np.arange(dx / 2 - G * dx + x_min, x_min + Lx + G * dx, dx)
y = np.arange(dy / 2 - G * dy + y_min, y_min + Ly + G * dy, dy)
X, Y = np.meshgrid(x, y, indexing="ij")

fig, ax = plt.subplots()
c = ax.contourf(X, Y, ni, cmap="jet", levels=50)
plt.colorbar(c)
plt.title("$n_i$")
if is_include_circle:
    circle = Wedge(
        center=(-0.5, 0),
        r=0.1,
        theta1=0,
        theta2=360,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_patch(circle)
ax.set_xlim(x_min, x_min + Lx)
ax.set_ylim(y_min, y_min + Ly)
ax.set_xlabel("$x/L_x$")
ax.set_ylabel("$y/L_y$")
plt.tight_layout()
plt.savefig(f"{file_path}/number_density.png")

fig, ax = plt.subplots()
c = ax.contourf(X, Y, phi, cmap="jet", levels=50)
plt.colorbar(c)
plt.title("$e\\phi/2k_BT_i$")
if is_include_circle:
    circle = Wedge(
        center=(-0.5, 0),
        r=0.1,
        theta1=0,
        theta2=360,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_patch(circle)
ax.set_xlim(x_min, x_min + Lx)
ax.set_ylim(y_min, y_min + Ly)
ax.set_xlabel("$x/L_x$")
ax.set_ylabel("$y/L_y$")
plt.tight_layout()
plt.savefig(f"{file_path}/potential.png")

plt.figure()
plt.plot(x, phi[:, G], "o-", label="$y/L_y=-1$")
plt.plot(x, phi[:, phi.shape[1] // 2], "o-", label="$y/L_y=0$")
plt.xlabel("$x/L_x$")
plt.ylabel("$\\phi$")
plt.legend()
plt.tight_layout()
plt.savefig(f"{file_path}/potential_profiles.png")

plt.show()

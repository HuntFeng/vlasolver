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

n = 256
nx, ny = 2 * n, n
Lx, Ly = 1, 0.5
G = 3
is_include_circle = True

file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/output_{n}_512.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)

dx = Lx / nx
dy = Ly / ny
x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
X, Y = np.meshgrid(x, y, indexing="ij")

fig, ax = plt.subplots(figsize=(6, 3))
# c = ax.contourf(X, Y, ni, cmap="jet", levels=50)
c = ax.pcolormesh(X, Y, ni, cmap="jet")
plt.colorbar(c)
plt.title("$n$")
if is_include_circle:
    circle = Wedge(
        center=(0.375, 0),
        r=0.125,
        theta1=0,
        theta2=180,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_patch(circle)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.5)
ax.set_xlabel("$x/L_x$")
ax.set_ylabel("$y/L_x$")
plt.tight_layout()
# plt.savefig(f"{file_path}/density.svg")

plt.figure()
plt.plot(x, ni[:, G], "-", label="$y=0$")
plt.plot(x, ni[:, ny // 2 + G], "-", label="$y=0.25$")
plt.xlabel("$x/L_x$")
plt.ylabel("$n$")
plt.legend()
# plt.savefig(f"{file_path}/electric_field_profiles.svg")
plt.show()

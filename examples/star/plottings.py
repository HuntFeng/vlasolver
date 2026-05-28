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

nx, ny = 128, 128
Lx, Ly = 1, 1
G = 3
is_include_star = True

step = 1000
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/star/output_{step:04d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)

dx = Lx / nx
dy = Ly / ny
x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
X, Y = np.meshgrid(x, y, indexing="ij")

def surface(x, y):
    x0 = 0.5
    y0 = 0.5
    rr = np.sqrt(np.pow(x - x0, 2) + np.pow(y - y0, 2));
    ang = np.atan2(y - y0, x - x0);
    return rr - (0.15 + 0.04 * np.sin(4 * ang));

fig, ax = plt.subplots()
c = ax.contourf(X, Y, ni, cmap="jet", levels=50)
plt.colorbar(c)
plt.title("$n_i$")
if is_include_star:
    ax.contourf(X, Y, surface(X,Y), levels=[-100, 0], colors='white')
    ax.contour(X, Y, surface(X,Y), levels=[0], colors='black', linewidths=2)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xlabel("$x/L_x$")
ax.set_ylabel("$y/L_x$")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{file_path}/number_density.png")

fig, ax = plt.subplots()
c = ax.contourf(X, Y, phi, cmap="jet", levels=50)
plt.colorbar(c)
plt.title("$e\\phi/2k_BT_i$")
if is_include_star:
    ax.contourf(X, Y, surface(X,Y), levels=[-100, 0], colors='white')
    ax.contour(X, Y, surface(X,Y), levels=[0], colors='black', linewidths=2)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_xlabel("$x/L_x$")
ax.set_ylabel("$y/L_x$")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{file_path}/potential.png")

plt.show()

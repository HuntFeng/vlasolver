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

nx, ny = 160, 50
Lx, Ly = 1, 0.5
G = 3
step = 300
with h5py.File(f"data/plasma_past_charged_cylinder/output_{step:03d}.h5", "r") as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)

ni = ni[G:-G, G:-G]
Ex = Ex[G:-G, G:-G]
phi = phi[G:-G, G:-G]

dx = Lx / nx
dy = Ly / ny
x = np.arange(dx / 2, Lx, dx)
y = np.arange(dy / 2, Ly, dy)
Y, X = np.meshgrid(y, x)

fig, ax = plt.subplots(figsize=(6, 3))
c = ax.contourf(X, Y, ni / ni.max(), cmap="jet", levels=50)
plt.colorbar(c, label="$n_i$")
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
ax.set_xlim(dx / 2, Lx - dx / 2)
ax.set_ylim(dy / 2, Ly - dy / 2)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")


fig, ax = plt.subplots(figsize=(6, 3))
c = ax.contourf(X, Y, phi, cmap="jet", levels=50)
plt.colorbar(c, label="$\\phi$")
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
ax.set_xlim(dx / 2, Lx - dx / 2)
ax.set_ylim(dy / 2, Ly - dy / 2)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

fig, ax = plt.subplots(figsize=(6, 3))
c = ax.contourf(X, Y, Ex, cmap="jet", levels=50)
plt.colorbar(c, label="$E_x$")
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
ax.set_xlim(dx / 2, Lx - dx / 2)
ax.set_ylim(dy / 2, Ly - dy / 2)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")


plt.figure()
plt.plot(x, Ex[:, G], "o-", label="$y=0$")
plt.plot(x, Ex[:, -G], "o-", label="$y=0.5$")
plt.xlabel("$x$")
plt.ylabel("$E_x$")
plt.legend()

plt.figure()
plt.plot(x, phi[:, G], "o-", label="$y=0$")
plt.plot(x, phi[:, -G], "o-", label="$y=0.5$")
plt.xlabel("$x$")
plt.ylabel("$\\phi$")
plt.legend()


plt.show()

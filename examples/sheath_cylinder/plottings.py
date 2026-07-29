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
x_min, y_min = -10, -10
Lx, Ly = 20, 20
G = 3
is_include_circle = True

step = 4000
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/sheath_cylinder/output_{step:04d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
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
        center=(0, 0),
        r=1,
        theta1=0,
        theta2=360,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_patch(circle)
ax.set_xlim(x_min, x_min + Lx)
ax.set_ylim(y_min, y_min + Ly)
ax.set_xlabel("$x/\\lambda_D$")
ax.set_ylabel("$y/\\lambda_D$")
plt.tight_layout()
plt.savefig(f"{file_path}/number_density.png")

fig, ax = plt.subplots()
c = ax.contourf(X, Y, ni - ne, cmap="jet", levels=50)
plt.colorbar(c)
plt.title("$\\rho$")
if is_include_circle:
    circle = Wedge(
        center=(0, 0),
        r=1,
        theta1=0,
        theta2=360,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_patch(circle)
ax.set_xlim(x_min, x_min + Lx)
ax.set_ylim(y_min, y_min + Ly)
ax.set_xlabel("$x/\\lambda_D$")
ax.set_ylabel("$y/\\lambda_D$")
plt.tight_layout()
plt.savefig(f"{file_path}/charge_density.png")

fig, ax = plt.subplots()
c = ax.contourf(X, Y, phi, cmap="jet", levels=50)
plt.colorbar(c)
plt.title("$e\\phi/2k_BT_e$")
if is_include_circle:
    circle = Wedge(
        center=(0, 0),
        r=1,
        theta1=0,
        theta2=360,
        facecolor="white",
        edgecolor="k",
        linewidth=2,
    )
    ax.add_patch(circle)
ax.set_xlim(x_min, x_min + Lx)
ax.set_ylim(y_min, y_min + Ly)
ax.set_xlabel("$x/\\lambda_D$")
ax.set_ylabel("$y/\\lambda_D$")
plt.tight_layout()
plt.savefig(f"{file_path}/potential.png")

plt.figure()
plt.plot(x, phi[:, phi.shape[1] // 2], "o-", label="$y/\\lambda_D=0$")
# phi_d = -np.log(np.sqrt(mi / me / (2 * np.pi))) / 2
from scipy.optimize import fsolve

mr = 100  # mi / me
Tr = 1  # Ti / Te
solution = fsolve(lambda phi: np.sqrt(mr / Tr) * np.exp(phi) + phi / Tr - 1, -1.0)
phi_d = solution[0]
print(phi_d)
phi_yukawa = phi_d / x * np.exp(-(x - 1))
plt.plot(x[x > 1], phi_yukawa[x > 1], "--", label="theory")
plt.xlabel("$x/\\lambda_D$")
plt.ylabel("$e\\phi / 2k_BT_e$")
plt.legend()
plt.tight_layout()
plt.savefig(f"{file_path}/potential_profiles.png")

plt.show()

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from oml_cylinder import OMLParameters, solve_oml
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
x_min, y_min = -20, -20
Lx, Ly = 40, 40
G = 3
is_include_circle = True

step = 12000
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/sheath_cylinder/output_{step:05d}.h5",
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
plt.title("$e\\phi/k_BT_e$")
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

params = OMLParameters(
    a=1.0,
    rmax=x.max(),
    Tr=1,
    mr=100,  # mi / me
)
sol = solve_oml(params)

xx = np.linspace(
    sol.params.a,
    sol.params.rmax,
    1000,
)
phi_a = sol.sol(xx)[0]
plt.figure()
plt.plot(x, phi[:, phi.shape[1] // 2], "o")
plt.plot(xx, phi_a, "-", label="theory")
plt.xlabel("$r/\\lambda_D$")
plt.ylabel("$e\\phi / k_BT_e$")
plt.axvline(x=1, color="k", linestyle="--", label="$r=1$")
plt.legend()
plt.xlim(0, x.max())
plt.tight_layout()
plt.savefig(f"{file_path}/potential_profiles.png")

# plt.figure()
# line_i = plt.plot(x, ni[:, ni.shape[1] // 2], "o", label="$n_i$")[0]
# line_e = plt.plot(x, ne[:, ne.shape[1] // 2], "o", label="$n_e$")[0]
# plt.plot(xx, sol.density(xx)[0], "-", color=line_i.get_color(), label="$n_{ia}$")
# plt.plot(xx, sol.density(xx)[1], "-", color=line_e.get_color(), label="$n_{ea}$")
# plt.xlabel("$r/\\lambda_D$")
# plt.ylabel("$n_i$")
# plt.axvline(x=1, color="k", linestyle="--", label="$r=1$")
# plt.legend()
# plt.xlim(0, xx.max())
# plt.savefig(f"{file_path}/density.png")

plt.show()

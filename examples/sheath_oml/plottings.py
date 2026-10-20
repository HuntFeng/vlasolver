import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from sheath_oml import OMLParameters, solve_oml
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

# quarter domain [0, 20]^2, cylinder at the origin (see main.cpp)
nx, ny = 64, 64
x_min, y_min = 0, 0
Lx, Ly = 20, 20
G = 3
is_include_circle = True

step = 12000
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/sheath_oml_quarter/output_{step:05d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    ni_bound = f["VTKHDF/CellData/ni_bound"][:].reshape(nx + 2 * G, ny + 2 * G)

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
        theta2=90,
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
        theta2=90,
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
        theta2=90,
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

# Radial cut along +x through the first interior row (y = dy/2), interior cells
# only (ghost cells hold boundary data, e.g. the half-Maxwellian injection).
jc = G
cut = (x > 0) & (x < x_min + Lx)
r_cut = np.hypot(x[cut], y[jc])

params = OMLParameters(
    a=1.0,
    rmax=x_min + Lx,  # grounded wall where the cut meets the domain boundary
    Tr=1,
    mr=100,  # mi / me
    outer_bc="dirichlet",
)
sol = solve_oml(params)

xx = np.linspace(
    sol.params.a,
    sol.params.rmax,
    1000,
)
phi_a = sol.sol(xx)[0]
plt.figure()
plt.plot(r_cut, phi[cut, jc], "o", fillstyle="none")
plt.plot(xx, phi_a, "-", label="theory")
plt.xlabel("$r/\\lambda_D$")
plt.ylabel("$e\\phi / k_BT_e$")
plt.axvline(x=1, color="k", linestyle="--", label="$r/\\lambda_D=1$")
plt.legend()
plt.xlim(0, sol.params.rmax)
plt.tight_layout()
plt.savefig(f"{file_path}/potential_profiles.png")

plt.figure()
line_i = plt.plot(r_cut, ni[cut, jc], "o", fillstyle="none", label="$n_i$")[0]
line_e = plt.plot(r_cut, ne[cut, jc], "o", fillstyle="none", label="$n_e$")[0]
line_b = plt.plot(r_cut, ni_bound[cut, jc], "s", fillstyle="none", label="$n_{i,bound}$")[0]
plt.plot(
    r_cut,
    ni[cut, jc] - ni_bound[cut, jc],
    "x",
    color=line_i.get_color(),
    label="$n_i - n_{i,bound}$",
)
plt.plot(xx, sol.density(xx)[0], "-", color=line_i.get_color(), label="$n_{ia}$")
plt.plot(xx, sol.density(xx)[1], "-", color=line_e.get_color(), label="$n_{ea}$")
plt.xlabel("$r/\\lambda_D$")
plt.ylabel("$n_i$")
plt.axvline(x=1, color="k", linestyle="--", label="$r=1$")
plt.legend()
plt.xlim(0, sol.params.rmax)
plt.savefig(f"{file_path}/density_profiles.png")

plt.show()

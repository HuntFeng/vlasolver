import configparser
import glob
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

# Due to the initial unbalance, the potential rings at the box-scale 
# electron plasma oscillation (period ~6 / omega_pe, decay time ~70 / omega_pe), 
# All fields below are time averaged to save simulation time
# (None = last output). Skip the initial transient, and cover several periods.
data_folder = "sheath_oml"
avg_start_step = 20
avg_end_step = None
ringing_period = 2 * np.pi  # ~1 / omega_pe, for reporting the window length only

file_path = os.path.dirname(os.path.realpath(__file__))
config = configparser.ConfigParser(inline_comment_prefixes=(";",))
config.read(f"{file_path}/input.ini")
dt = config.getfloat("world", "dt")

# the writer zero-pads the step to the width of total_steps, so match any padding
outputs = {}
for path in glob.glob(f"{file_path}/../../data/{data_folder}/output_*.h5"):
    outputs[int(os.path.basename(path)[len("output_") : -len(".h5")])] = path
steps = sorted(
    s for s in outputs if s >= avg_start_step and (avg_end_step is None or s <= avg_end_step)
)
if not steps:
    raise FileNotFoundError(f"no outputs in data/{data_folder} within the averaging window")

fields = {"ni": 0.0, "ne": 0.0, "phi": 0.0, "ni_bound": 0.0}
for s in steps:
    with h5py.File(outputs[s], "r") as f:
        for k in fields:
            fields[k] = fields[k] + f[f"VTKHDF/CellData/{k}"][:].reshape(nx + 2 * G, ny + 2 * G)
ni, ne, phi, ni_bound = (fields[k] / len(steps) for k in ["ni", "ne", "phi", "ni_bound"])

t_window = (steps[-1] - steps[0]) * dt
print(
    f"Averaging {len(steps)} outputs, steps {steps[0]}-{steps[-1]} "
    f"(t = {steps[0] * dt:g}-{steps[-1] * dt:g}, {t_window / ringing_period:.1f} ringing periods)"
)
avg_label = f"$\\langle\\cdot\\rangle_t$, $t\\in[{steps[0] * dt:g}, {steps[-1] * dt:g}]$"

dx = Lx / nx
dy = Ly / ny
x = np.arange(dx / 2 - G * dx + x_min, x_min + Lx + G * dx, dx)
y = np.arange(dy / 2 - G * dy + y_min, y_min + Ly + G * dy, dy)
X, Y = np.meshgrid(x, y, indexing="ij")

fig, ax = plt.subplots()
c = ax.contourf(X, Y, ni, cmap="jet", levels=50)
plt.colorbar(c)
plt.title(f"$n_i$, {avg_label}", fontsize=12)
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
plt.title(f"$\\rho$, {avg_label}", fontsize=12)
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
plt.title(f"$e\\phi/k_BT_e$, {avg_label}", fontsize=12)
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
    rmax=x_min + 40,  # grounded wall where the cut meets the domain boundary
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
plt.plot(r_cut, phi[cut, jc], "o", fillstyle="none", label="simulation, time avg")
plt.plot(xx, phi_a, "-", label="theory")
plt.xlabel("$r/\\lambda_D$")
plt.ylabel("$e\\phi / k_BT_e$")
plt.axvline(x=1, color="k", linestyle="--", label="$r/\\lambda_D=1$")
plt.legend()
plt.xlim(0, Lx)
plt.tight_layout()
plt.savefig(f"{file_path}/potential_profiles.png")

plt.figure()
line_i = plt.plot(r_cut, ni[cut, jc], "o", fillstyle="none", label="$n_i$")[0]
line_e = plt.plot(r_cut, ne[cut, jc], "o", fillstyle="none", label="$n_e$")[0]
# line_b = plt.plot(r_cut, ni_bound[cut, jc], "s", fillstyle="none", label="$n_{i,bound}$")[0]
# plt.plot(
#     r_cut,
#     ni[cut, jc] - ni_bound[cut, jc],
#     "x",
#     color=line_i.get_color(),
#     label="$n_i - n_{i,bound}$",
# )
plt.plot(xx, sol.density(xx)[0], "-", color=line_i.get_color(), label="$n_{ia}$")
plt.plot(xx, sol.density(xx)[1], "-", color=line_e.get_color(), label="$n_{ea}$")
plt.xlabel("$r/\\lambda_D$")
plt.ylabel("$n_i$")
plt.axvline(x=1, color="k", linestyle="--", label="$r=1$")
plt.legend()
plt.xlim(0, Lx)
plt.savefig(f"{file_path}/density_profiles.png")

plt.show()

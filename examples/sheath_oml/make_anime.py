import os

import h5py
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sheath_oml import OMLParameters, solve_oml
from tqdm import tqdm

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 16,
    }
)

# Animation parameters
total_steps = 2000
frame_interval = 20
start_step = 0

# Grid parameters
# quarter domain [0, 20]^2, cylinder at the origin (see main.cpp)
nx, ny = 64, 64
Lx, Ly = 20, 20
x_min, y_min = 0, 0
G = 3
is_include_circle = True

file_path = os.path.dirname(os.path.realpath(__file__))

# Create coordinate arrays
dx, dy = Lx / nx, Ly / ny
x = np.arange(dx / 2 - G * dx + x_min, Lx + G * dx + x_min, dx)
y = np.arange(dy / 2 - G * dy + y_min, Ly + G * dy + y_min, dy)
X, Y = np.meshgrid(x, y, indexing="ij")


def surface(x, y):
    return x**2 + y**2 - 1.0**2


def load_data(step):
    try:
        with h5py.File(
            f"{file_path}/../../data/sheath_oml/output_{step:04d}.h5", "r"
        ) as f:
            ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
            ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
            phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
        return ni, ne, phi
    except FileNotFoundError:
        return None, None, None


def draw_circle(ax):
    ax.contourf(X, Y, surface(X, Y), levels=[-100, 0], colors="white")
    ax.contour(X, Y, surface(X, Y), levels=[0], colors="black", linewidths=2)


# Radial cut along +x through the first interior row (y = dy/2), interior cells
# only (matching plottings.py)
jc = G
cut = (x > 0) & (x < x_min + Lx)
r_cut = np.hypot(x[cut], y[jc])

# Analytical OML profiles (same parameters as plottings.py)
sol = solve_oml(
    OMLParameters(
        a=1.0,
        rmax=x_min + 40,  # grounded wall where the cut meets the domain boundary
        Tr=1,
        mr=100,  # mi / me
        outer_bc="dirichlet",
    )
)
r_a = np.linspace(sol.params.a, sol.params.rmax, 1000)
phi_a = sol.sol(r_a)[0]
ni_a, ne_a = sol.density(r_a)

# Set up figure and subplots
fig, ax = plt.subplots(2, 2, figsize=(14, 12))

# Preload first frame to create colorbars once
ni0, ne0, phi0 = load_data(start_step)
if ni0 is None:
    raise FileNotFoundError(f"No data found at step {start_step}")

ni_norm = mcolors.Normalize(vmin=0, vmax=ni0.max())
phi_norm = mcolors.Normalize(vmin=-2.0, vmax=0.0)


def draw_contours(ni, phi, step):
    ax[0, 0].contourf(X, Y, ni, cmap="jet", levels=50, norm=ni_norm)
    if is_include_circle:
        draw_circle(ax[0, 0])
    ax[0, 0].set_title(f"$n_i$ (Step: {step})")

    ax[1, 0].contourf(X, Y, phi, cmap="jet", levels=50, norm=phi_norm)
    if is_include_circle:
        draw_circle(ax[1, 0])
    ax[1, 0].set_title(f"$e\\phi/k_BT_e$ (Step: {step})")


draw_contours(ni0, phi0, start_step)
cb1 = fig.colorbar(cm.ScalarMappable(norm=ni_norm, cmap="jet"), ax=ax[0, 0])
cb2 = fig.colorbar(cm.ScalarMappable(norm=phi_norm, cmap="jet"), ax=ax[1, 0])
for _ax in ax[:, 0]:
    _ax.set_xlim(x_min, x_min + Lx)
    _ax.set_ylim(y_min, y_min + Ly)
    _ax.set_xlabel("$x/\\lambda_D$")
    _ax.set_ylabel("$y/\\lambda_D$")
    _ax.set_aspect("equal")

# Number density profile line plot
(line_ni,) = ax[0, 1].plot(r_cut, ni0[cut, jc], "o", fillstyle="none", label="$n_i$")
(line_ne,) = ax[0, 1].plot(r_cut, ne0[cut, jc], "o", fillstyle="none", label="$n_e$")
ax[0, 1].plot(r_a, ni_a, "-", color=line_ni.get_color(), label="$n_{ia}$ (theory)")
ax[0, 1].plot(r_a, ne_a, "-", color=line_ne.get_color(), label="$n_{ea}$ (theory)")
ax[0, 1].axvline(x=1, color="k", linestyle="--")
ax[0, 1].set_xlabel("$r/\\lambda_D$")
ax[0, 1].set_ylabel("$n$")
ax[0, 1].set_title(f"Density Profile (Step: {start_step})")
ax[0, 1].legend(loc="lower right")
ax[0, 1].set_xlim(x_min, x_min + Lx)
ax[0, 1].set_ylim(0.0, 1.2)
ax[0, 1].grid(True, alpha=0.3)

# Potential profile line plot
(line_phi,) = ax[1, 1].plot(
    r_cut, phi0[cut, jc], "o", fillstyle="none", label="simulation"
)
ax[1, 1].plot(r_a, phi_a, "-", color=line_phi.get_color(), label="theory")
ax[1, 1].axvline(x=1, color="k", linestyle="--")
ax[1, 1].set_xlabel("$r/\\lambda_D$")
ax[1, 1].set_ylabel("$e\\phi/k_BT_e$")
ax[1, 1].set_title(f"Potential Profile (Step: {start_step})")
ax[1, 1].legend(loc="lower right")
ax[1, 1].set_xlim(x_min, x_min + Lx)
ax[1, 1].set_ylim(-2.0, 1.0)
ax[1, 1].grid(True, alpha=0.3)

fig.tight_layout()


def animate(frame):
    current_step = start_step + frame * frame_interval
    ni, ne, phi = load_data(current_step)

    if ni is None:
        return []

    # Remove old contour collections from the 2D axes only (keep line plots)
    for _ax in ax[:, 0]:
        for coll in list(_ax.collections):
            coll.remove()

    draw_contours(ni, phi, current_step)

    line_ni.set_ydata(ni[cut, jc])
    line_ne.set_ydata(ne[cut, jc])
    ax[0, 1].set_title(f"Density Profile (Step: {current_step})")

    line_phi.set_ydata(phi[cut, jc])
    ax[1, 1].set_title(f"Potential Profile (Step: {current_step})")

    return ()


num_frames = (total_steps - start_step) // frame_interval + 1

anim = animation.FuncAnimation(
    fig, animate, frames=num_frames, interval=100, repeat=True, blit=False
)

with tqdm(total=num_frames, desc="Rendering animation") as pbar:
    anim.save(
        f"{file_path}/sheath_cylinder.mp4",
        progress_callback=lambda i, n: pbar.update(1),
    )
plt.show()

import os

import h5py
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
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
total_steps = 12000
frame_interval = 120
start_step = 0

# Grid parameters
nx, ny = 128, 128
Lx, Ly = 40, 40
x_min, y_min = -20, -20
G = 3
is_include_circle = True

file_path = os.path.dirname(os.path.realpath(__file__))

# Create coordinate arrays
dx, dy = Lx / nx, Ly / ny
x = np.arange(dx / 2 - G * dx + x_min, Lx + G * dx + x_min, dx)
y = np.arange(dy / 2 - G * dy + y_min, Ly + G * dy + y_min, dy)
X, Y = np.meshgrid(x, y, indexing="ij")


def surface(x, y):
    return x**2 + y**2 - 0.1**2


def load_data(step):
    try:
        with h5py.File(
            f"{file_path}/../../data/sheath_cylinder/output_{step:05d}.h5", "r"
        ) as f:
            ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
            phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
        return ni, phi
    except FileNotFoundError:
        return None, None, None


def draw_circle(ax):
    ax.contourf(X, Y, surface(X, Y), levels=[-100, 0], colors="white")
    ax.contour(X, Y, surface(X, Y), levels=[0], colors="black", linewidths=2)


# Potential profile slice indices (matching plottings.py)
idx_y_neg1 = G  # y ≈ -1 (bottom edge)
idx_y_0 = (ny + 2 * G) // 2  # y ≈ 0 (mid-plane)

# Set up figure and subplots
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Preload first frame to create colorbars once
ni0, phi0 = load_data(start_step)
if ni0 is None:
    raise FileNotFoundError(f"No data found at step {start_step}")

ni_norm = mcolors.Normalize(vmin=0, vmax=ni0.max())
phi_norm = mcolors.Normalize(vmin=-2.0, vmax=0.0)

ax[0].contourf(X, Y, ni0, cmap="jet", levels=50)
cb1 = fig.colorbar(cm.ScalarMappable(norm=ni_norm, cmap="jet"), ax=ax[0])
if is_include_circle:
    draw_circle(ax[0])
ax[0].set_xlim(x_min, x_min + Lx)
ax[0].set_ylim(y_min, y_min + Ly)
ax[0].set_xlabel("$x/L_x$")
ax[0].set_ylabel("$y/L_y$")
ax[0].set_aspect("equal")

c2 = ax[1].contourf(X, Y, phi0, cmap="jet", levels=50, norm=phi_norm)
cb2 = fig.colorbar(cm.ScalarMappable(norm=phi_norm, cmap="jet"), ax=ax[1])
if is_include_circle:
    draw_circle(ax[1])
ax[1].set_xlim(x_min, x_min + Lx)
ax[1].set_ylim(y_min, y_min + Ly)
ax[1].set_xlabel("$x/L_x$")
ax[1].set_ylabel("$y/L_y$")
ax[1].set_aspect("equal")

# Potential profile line plot
(line_0,) = ax[2].plot(x, phi0[:, idx_y_0], "o-", label="$y/L_y=0$")
ax[2].set_xlabel("$x/L_x$")
ax[2].set_ylabel("$e\\phi/2k_BT_i$")
ax[2].set_title(f"Potential Profile (Step: {start_step})")
ax[2].legend()
ax[2].set_xlim(x_min, x_min + Lx)
ax[2].set_ylim(-2.0, 1.0)
ax[2].grid(True, alpha=0.3)

fig.tight_layout()


def animate(frame):
    current_step = start_step + frame * frame_interval
    ni, phi = load_data(current_step)

    if ni is None:
        return []

    # Remove old collections from data axes only (colorbar axes are untouched)
    for _ax in ax.flatten():
        for coll in list(_ax.collections):
            coll.remove()

    ax[0].contourf(X, Y, ni, cmap="jet", levels=50, norm=ni_norm)
    if is_include_circle:
        draw_circle(ax[0])
    ax[0].set_title(f"$n_i$ (Step: {current_step})")

    ax[1].contourf(X, Y, phi, cmap="jet", levels=50, norm=phi_norm)
    if is_include_circle:
        draw_circle(ax[1])
    ax[1].set_title(f"$e\\phi/2k_BT_i$ (Step: {current_step})")

    line_0.set_ydata(phi[:, idx_y_0])
    ax[2].set_title(f"Potential Profile (Step: {current_step})")

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

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
total_steps = 1000
frame_interval = 5
start_step = 0

# Grid parameters
nx, ny = 128, 128
Lx, Ly = 1, 1
G = 3
is_include_star = True

file_path = os.path.dirname(os.path.realpath(__file__))

# Create coordinate arrays
dx, dy = Lx / nx, Ly / ny
x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
X, Y = np.meshgrid(x, y, indexing="ij")


def surface(x, y):
    x0, y0 = 0.5, 0.5
    rr = np.sqrt(np.power(x - x0, 2) + np.power(y - y0, 2))
    ang = np.arctan2(y - y0, x - x0)
    return rr - (0.15 + 0.04 * np.sin(4 * ang))


def load_data(step):
    try:
        with h5py.File(
            f"{file_path}/../../data/star/output_{step:04d}.h5", "r"
        ) as f:
            ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
            phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
            Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
        return ni, phi, Ex
    except FileNotFoundError:
        return None, None, None


def draw_star(ax):
    ax.contourf(X, Y, surface(X, Y), levels=[-100, 0], colors="white")
    ax.contour(X, Y, surface(X, Y), levels=[0], colors="black", linewidths=2)


# Set up figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Preload first frame to create colorbars once
ni0, phi0, Ex0 = load_data(start_step)
if ni0 is None:
    raise FileNotFoundError(f"No data found at step {start_step}")

ni_norm = mcolors.Normalize(vmin=0, vmax=4.5)
phi_norm = mcolors.Normalize(vmin=phi0.min(), vmax=phi0.max())
Ex_norm = mcolors.Normalize(vmin=-350, vmax=500)

ax1.contourf(X, Y, ni0, cmap="jet", levels=50, norm=ni_norm)
cb1 = fig.colorbar(cm.ScalarMappable(norm=ni_norm, cmap="jet"), ax=ax1)
if is_include_star:
    draw_star(ax1)
ax1.set_xlim(0, Lx)
ax1.set_ylim(0, Ly)
ax1.set_xlabel("$x/L_x$")
ax1.set_ylabel("$y/L_x$")
ax1.set_aspect("equal")

c2 = ax2.contourf(X, Y, phi0, cmap="jet", levels=50, norm=phi_norm)
cb2 = fig.colorbar(cm.ScalarMappable(norm=phi_norm, cmap="jet"), ax=ax2)
if is_include_star:
    draw_star(ax2)
ax2.set_xlim(0, Lx)
ax2.set_ylim(0, Ly)
ax2.set_xlabel("$x/L_x$")
ax2.set_ylabel("$y/L_x$")
ax2.set_aspect("equal")

c3 = ax3.contourf(X, Y, Ex0, cmap="jet", levels=50, norm=Ex_norm)
cb3 = fig.colorbar(cm.ScalarMappable(norm=Ex_norm, cmap="jet"), ax=ax3)
if is_include_star:
    draw_star(ax3)
ax3.set_xlim(0, Lx)
ax3.set_ylim(0, Ly)
ax3.set_xlabel("$x/L_x$")
ax3.set_ylabel("$y/L_x$")
ax3.set_aspect("equal")

fig.tight_layout()


def animate(frame):
    current_step = start_step + frame * frame_interval
    ni, phi, Ex = load_data(current_step)

    if ni is None:
        return []

    # Remove old collections from data axes only (colorbar axes are untouched)
    for ax in (ax1, ax2, ax3):
        for coll in list(ax.collections):
            coll.remove()

    ax1.contourf(X, Y, ni, cmap="jet", levels=50, norm=ni_norm)
    if is_include_star:
        draw_star(ax1)
    ax1.set_title(f"$n_i$ (Step: {current_step})")

    ax2.contourf(X, Y, phi, cmap="jet", levels=50, norm=phi_norm)
    if is_include_star:
        draw_star(ax2)
    ax2.set_title(f"$e\\phi/2k_BT_i$ (Step: {current_step})")

    ax3.contourf(X, Y, Ex, cmap="jet", levels=50, norm=Ex_norm)
    if is_include_star:
        draw_star(ax3)
    ax3.set_title(f"$E_x$ (Step: {current_step})")

    return ()


num_frames = (total_steps - start_step) // frame_interval + 1

anim = animation.FuncAnimation(
    fig, animate, frames=num_frames, interval=100, repeat=True, blit=False
)

with tqdm(total=num_frames, desc="Rendering animation") as pbar:
    anim.save(
        f"{file_path}/star.mp4",
        progress_callback=lambda i, n: pbar.update(1),
    )
plt.show()

import os

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

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
is_include_star = False

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
        return ni, phi
    except FileNotFoundError:
        return None, None


# Set up the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


def animate(frame):
    current_step = start_step + frame * frame_interval

    ni, phi = load_data(current_step)

    if ni is None or phi is None:
        return []

    ax1.clear()
    ax2.clear()

    ax1.contourf(X, Y, ni, cmap="jet", levels=50)
    if is_include_star:
        ax1.contourf(X, Y, surface(X, Y), levels=[-100, 0], colors="white")
        ax1.contour(X, Y, surface(X, Y), levels=[0], colors="black", linewidths=2)
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, Ly)
    ax1.set_xlabel("$x/L_x$")
    ax1.set_ylabel("$y/L_x$")
    ax1.set_aspect("equal")
    ax1.set_title(f"$n_i$ (Step: {current_step})")

    ax2.contourf(X, Y, phi, cmap="jet", levels=50)
    if is_include_star:
        ax2.contourf(X, Y, surface(X, Y), levels=[-100, 0], colors="white")
        ax2.contour(X, Y, surface(X, Y), levels=[0], colors="black", linewidths=2)
    ax2.set_xlim(0, Lx)
    ax2.set_ylim(0, Ly)
    ax2.set_xlabel("$x/L_x$")
    ax2.set_ylabel("$y/L_x$")
    ax2.set_aspect("equal")
    ax2.set_title(f"$e\\phi/2k_BT_i$ (Step: {current_step})")

    fig.tight_layout()
    return []


num_frames = (total_steps - start_step) // frame_interval + 1

anim = animation.FuncAnimation(
    fig, animate, frames=num_frames, interval=100, repeat=True
)

anim.save(f"{file_path}/star.mp4")
plt.show()

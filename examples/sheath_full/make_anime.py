import os

import h5py
import matplotlib.animation as animation
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

# Animation parameters
total_steps = 8000  # User can modify this
frame_interval = 80  # User can modify this (step interval between frames)
start_step = 0  # Starting step

# Grid parameters
nx, ny, nvx, nvy = 10, 125, 30, 110
x_min, y_min = 0, 0
Lx, Ly = 1, 20

vx_min_e, vy_min_e = -4, -5
Lvx_e, Lvy_e = 8, 10
# in calculation, the ranges are devided by vr
# vx_min_i, vy_min_i = -4, -8
# Lvx_i, Lvy_i = 8, 9
vx_min_i, vy_min_i = -4, -20
Lvx_i, Lvy_i = 8, 21
G = 3
is_include_ghost = False


def load_data(step):
    """Load data for a given step"""
    try:
        with h5py.File(
            f"{os.path.dirname(os.path.realpath(__file__))}/../../data/sheath_full/output_{step:04d}.h5",
            "r",
        ) as f:
            ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
            ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
            phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)

        if not is_include_ghost:
            ni = ni[G:-G, G:-G]
            ne = ne[G:-G, G:-G]
            phi = phi[G:-G, G:-G]

        return ni, ne, phi
    except FileNotFoundError:
        return None, None, None


# Create coordinate arrays
if is_include_ghost:
    dx, dy = Lx / nx, Ly / ny
    x = np.arange(x_min - G * dx + dx / 2, x_min + Lx + G * dx, dx)
    y = np.arange(y_min - G * dy + dy / 2, y_min + Ly + G * dy, dy)
    dvx_e, dvy_e = Lvx_e / nvx, Lvy_e / nvy
    vx_e = np.arange(
        vx_min_e - G * dvx_e + dvx_e / 2, vx_min_e + Lvx_e + G * dvx_e, dvx_e
    )
    vy_e = np.arange(
        vy_min_e - G * dvy_e + dvy_e / 2, vy_min_e + Lvy_e + G * dvy_e, dvy_e
    )
    dvx_i, dvy_i = Lvx_i / nvx, Lvy_i / nvy
    vx_i = np.arange(
        vx_min_i - G * dvx_i + dvx_i / 2, vx_min_i + Lvx_i + G * dvx_i, dvx_i
    )
    vy_i = np.arange(
        vy_min_i - G * dvy_i + dvy_i / 2, vy_min_i + Lvy_i + G * dvy_i, dvy_i
    )
else:
    dx, dy = Lx / nx, Ly / ny
    x = np.arange(x_min + dx / 2, x_min + Lx, dx)
    y = np.arange(y_min + dy / 2, y_min + Ly, dy)

    dvx_e, dvy_e = Lvx_e / nvx, Lvy_e / nvy
    vx_e = np.arange(vx_min_e + dvx_e / 2, vx_min_e + Lvx_e, dvx_e)
    vy_e = np.arange(vy_min_e + dvy_e / 2, vy_min_e + Lvy_e, dvy_e)
    dvx_i, dvy_i = Lvx_i / nvx, Lvy_i / nvy
    vx_i = np.arange(vx_min_i + dvx_i / 2, vx_min_i + Lvx_i, dvx_i)
    vy_i = np.arange(vy_min_i + dvy_i / 2, vy_min_i + Lvy_i, dvy_i)

# Set up the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Initialize empty line objects
(line_phi,) = ax1.plot([], [], "b-", linewidth=2)
(line_ni,) = ax2.plot([], [], "r-", label="$n_i$", linewidth=2)
(line_ne,) = ax2.plot([], [], "b-", label="$n_e$", linewidth=2)

# Set up the axes
ax1.set_xlabel("$y$")
ax1.set_ylabel("$\\phi$")
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("$y$")
ax2.set_ylabel("$n$")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Initialize with first frame to set axis limits
ni_init, ne_init, phi_init = load_data(start_step)
if ni_init is not None and phi_init is not None:
    ax1.set_xlim(y.min(), y.max())
    ax1.set_ylim(-5.0, 1.0)

    ne_init = np.exp(phi_init[phi_init.shape[0] // 2, :])
    ni_slice_init = ni_init[ni_init.shape[0] // 2, :]
    ax2.set_xlim(y.min(), y.max())
    ax2.set_ylim(0.0, 1.2)


def animate(frame):
    """Animation function called for each frame"""
    current_step = start_step + frame * frame_interval

    ni, ne, phi = load_data(current_step)

    if ni is None or ne is None or phi is None:
        return line_phi, line_ni, line_ne

    # Update phi plot
    phi_slice = phi[phi.shape[0] // 2, :]
    line_phi.set_data(y, phi_slice)

    # Update density plots
    ni_slice = ni[ni.shape[0] // 2, :]
    ne_slice = ne[ne.shape[0] // 2, :]
    line_ni.set_data(y, ni_slice)
    line_ne.set_data(y, ne_slice)

    # Update titles with current step
    ax1.set_title(f"Electric Potential (Step: {current_step})")
    ax2.set_title(f"Density Profiles (Step: {current_step})")

    return line_phi, line_ni, line_ne


# Calculate number of frames
num_frames = (total_steps - start_step) // frame_interval + 1

# Create animation
anim = animation.FuncAnimation(
    fig, animate, frames=num_frames, interval=100, repeat=True
)

anim.save("sheath_floating_wall.mp4")

plt.tight_layout()
plt.show()

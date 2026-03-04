import os

import h5py
import matplotlib.animation as animation
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

# Animation parameters
total_steps = 1536  # User can modify this
frame_interval = 150  # User can modify this (step interval between frames)
start_step = 0  # Starting step
is_include_circle = True  # User can modify this (whether to include the circle)

# Grid parameters
nx, ny = 256 * 2, 256
x_min, y_min = 0, 0
Lx, Ly = 1, 0.5
dx, dy = Lx / nx, Ly / ny
G = 3
file_path = os.path.dirname(os.path.realpath(__file__))


def load_data(step):
    """Load data for a given step"""
    try:
        with h5py.File(
            f"{file_path}/../../data/advection/output_{ny}_{step:04d}.h5",
            "r",
        ) as f:
            ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
            ni = ni[G:-G, G:-G]

        return ni
    except FileNotFoundError:
        return None


# Create coordinate arrays
x = np.arange(x_min + dx / 2, x_min + Lx, dx)
y = np.arange(y_min + dy / 2, y_min + Ly, dy)
X, Y = np.meshgrid(x, y, indexing="ij")
ni = load_data(start_step)

# Set up the figure and subplots
fig, ax = plt.subplots()
pmesh = ax.pcolormesh(X, Y, ni, cmap="jet")
if is_include_circle:
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
ax.set_ylabel("$y/Lx$")
ax.set_xlabel("$x/Lx$")
fig.tight_layout()


def animate(frame):
    """Animation function called for each frame"""
    current_step = start_step + frame * frame_interval
    ni = load_data(current_step)
    if ni is None:
        return ni_plot
    pmesh.set_array(ni)
    ax.set_title(f"Density, Step: {current_step})")

    return (pmesh,)


# Calculate number of frames
num_frames = (total_steps - start_step) // frame_interval + 1

# Create animation
anim = animation.FuncAnimation(
    fig, animate, frames=num_frames, interval=100, repeat=True
)

anim.save(f"{file_path}/advection.mp4")

plt.show()

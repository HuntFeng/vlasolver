import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

nx, ny, nvx, nvy = 134, 70, 134, 134
Lx, Ly = 1, 0.5
ngc = 3
dx = Lx / (nx - 2 * ngc)
dy = Ly / (ny - 2 * ngc)
x = np.arange(dx / 2, Lx, dx)
y = np.arange(dy / 2, Ly, dy)
Y, X = np.meshgrid(y, x)

fig, ax = plt.subplots(figsize=(8, 4))
total_steps = 500
diag_step = 5
frames = total_steps // diag_step
# file = lambda step: f"data/debug_vlasov/output_{step:03d}.h5"
file = lambda step: f"data/plasma_past_charged_cylinder/output_{step:03d}.h5"
with h5py.File(file(0), "r") as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx, ny)

n = ni[ngc:-ngc, ngc:-ngc]
# n[(np.abs(X - 0.15) <= 0.04) & (np.abs(Y - 0.1) <= 0.1)] = np.nan
n[(X - 0.375) ** 2 + Y**2 <= 0.125**2] = np.nan
n_max = n.max() if n.max() > 0.0 else 1.0
im = ax.pcolormesh(X, Y, n / n_max, cmap="jet")
colorbar = fig.colorbar(im, ax=ax)


def animate(i):
    ax.clear()
    step = i * diag_step

    # Load the data for the current frame
    with h5py.File(file(step), "r") as f:
        ni = f["VTKHDF/CellData/ni"][:].reshape(nx, ny)

    n = ni[ngc:-ngc, ngc:-ngc]
    n_max = n.max() if n.max() > 0.0 else 1.0
    # n[(np.abs(X - 0.15) <= 0.04) & (np.abs(Y - 0.1) <= 0.1)] = np.nan
    n[(X - 0.375) ** 2 + Y**2 <= 0.125**2] = np.nan
    im = ax.pcolormesh(X, Y, n / n_max, cmap="jet")
    colorbar.update_normal(im)  # Update colorbar with new data
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"Ion Number Density (Step {step})")
    return [im]


# Create the animation
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
anim.save("number_density.mp4", writer="ffmpeg", fps=10)
plt.show()

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm

nx = ny = 256
Lx, Ly = 2.0, 2.0
G = 3

dx = Lx / nx
dy = Ly / ny
x = np.arange(-1.0 + dx / 2 - G * dx, 1.0 + dx / 2 + G * dx, dx)
y = np.arange(-1.0 + dy / 2 - G * dy, 1.0 + G * dy, dy)
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    # f"{file_path}/poisson_{ny}_0.h5",
    f"{file_path}/../../data/poisson/poisson_{ny}_0.h5",
    "r",
) as f:
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
    Ey = f["VTKHDF/CellData/Ey"][:].reshape(nx + 2 * G, ny + 2 * G)

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y, indexing="ij")
x0 = 0.02 * np.sqrt(5)
y0 = 0.02 * np.sqrt(3)

eps_safe = 1e-30
def surface(x, y):
    rr = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    ang = np.arctan2(y - y0, x - x0)
    return rr - (0.5 + 0.15 * np.sin(5 * ang))
Phi = surface(X, Y)
mask_minus = Phi < 0  # Omega^-: inside the irregular shape.

R2 = X**2 + Y**2
R2_safe = np.maximum(R2, eps_safe)

# u^+ - u^- and the two-sided u/grad fields, defined on the full grid.
u_minus = R2.copy()
u_plus = 0.1 * R2**2 - 0.01 * np.log(2.0 * np.sqrt(R2_safe))
u_exact = np.where(mask_minus, u_minus, u_plus)

fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
ax[0].plot_surface(X, Y, u_exact, edgecolor="black", cmap=cm.coolwarm)
ax[0].set_title("Example 4.2: exact")
ax[1].plot_surface(X, Y, phi, edgecolor="black", cmap=cm.coolwarm)
ax[1].set_title("Example 4.2: numerical")
plt.show()

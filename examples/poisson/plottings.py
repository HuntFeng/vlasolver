import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

nx = ny = 128
Lx, Ly = 1.0, 1.0
G = 3

dx = Lx / nx
dy = Ly / ny
x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/poisson/poisson_0.h5",
    "r",
) as f:
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
    Ey = f["VTKHDF/CellData/Ey"][:].reshape(nx + 2 * G, ny + 2 * G)

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y, indexing="ij")
mask = (X - 0.5) ** 2 + (Y - 0.5) ** 2 > 0.25**2
u_exact = np.exp(-(X**2 + Y**2))
u_exact[mask] = 0.0
dudx_exact = -2.0 * X * np.exp(-(X**2 + Y**2))
dudx_exact[mask] = 0.0
dudy_exact = -2.0 * Y * np.exp(-(X**2 + Y**2))
dudy_exact[mask] = 0.0


print(f"error phi = {np.linalg.norm((u_exact - phi)[G:-G, G:-G], np.inf)}")
print(f"error Ex = {np.linalg.norm((-dudx_exact - Ex)[G:-G, G:-G], np.inf)}")
print(f"error Ey = {np.linalg.norm((-dudy_exact - Ey)[G:-G, G:-G], np.inf)}")

fig = plt.figure()
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, phi, cmap="viridis")

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("$\\phi$")
ax1.set_title("$\\phi$")

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(X, Y, u_exact, cmap="viridis")

ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("$\\phi$")
ax2.set_title("$\\phi$ exact")


fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(221, projection="3d")
surf1 = ax1.plot_surface(X, Y, Ex, cmap="viridis")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("$E_x$")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
ax2 = fig.add_subplot(222, projection="3d")
surf2 = ax2.plot_surface(X, Y, Ey, cmap="viridis")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("$E_y$")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

ax3 = fig.add_subplot(223, projection="3d")
surf3 = ax3.plot_surface(X, Y, -dudx_exact, cmap="viridis")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("$E_x$ exact")
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

ax4 = fig.add_subplot(224, projection="3d")
surf4 = ax4.plot_surface(X, Y, -dudy_exact, cmap="viridis")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_title("$E_y$ exact")
fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

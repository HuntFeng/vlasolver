import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

nx, ny = 64, 64
Lx, Ly = 1.0, 1.0
G = 3

dx = Lx / nx
dy = Ly / ny
x = np.arange(dx / 2 - G * dx, Lx + G * dx , dx)
y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/poisson/poisson_0.h5",
    "r",
) as f:
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y, indexing='ij')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, phi, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Field Value')
plt.title('Poisson Field 3D Surface')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

nx, ny = 64, 64
dx, dy = 1.0 / nx, 1.0 / ny
x = np.arange(dx / 2, 1.0 + dx / 2, dx)
y = np.arange(dy / 2, 1.0 + dy / 2, dy)
X, Y = np.meshgrid(x, y, indexing="ij")
# source term
f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
n_total = (nx + 2) * (ny + 2)

# 1D Laplacian operators for full grid
Dx = diags([1, -2, 1], [-1, 0, 1], shape=(nx + 2, nx + 2)) / dx**2
Dy = diags([1, -2, 1], [-1, 0, 1], shape=(ny + 2, ny + 2)) / dy**2

# Identity matrices
Ix = identity(nx + 2)
Iy = identity(ny + 2)

# 2D Laplacian: A = Iy ⊗ Dx + Dy ⊗ Ix
A = (kron(Iy, Dx) + kron(Dy, Ix)).tolil()
plt.figure()
plt.spy(A)
plt.show()

# Right-hand side b
b = np.zeros(n_total)
# Interior indices: reshape to 2D, take [1:-1, 1:-1], flatten
interior_mask = np.zeros((nx + 2, ny + 2), dtype=bool)
interior_mask[1:-1, 1:-1] = True
b[interior_mask.flatten()] = f.flatten()

# Enforce Dirichlet BCs (u=0 on boundaries) by modifying boundary rows
boundary_mask = ~interior_mask
boundary_indices = np.where(boundary_mask.flatten())[0]
A[boundary_indices, :] = 0
A[boundary_indices, boundary_indices] = 1
b[boundary_mask.flatten()] = 0  # BC value = 0

# Solve Au = b
# u_flat = spsolve(A, b)

# Reshape to 2D
# u = u_flat.reshape((nx + 2, ny + 2))

import numpy as np
from scipy.sparse import diags, identity, kron, lil_matrix
from scipy.sparse.linalg import spsolve


def solve_poisson_2d(f, dx, dy):
    """
    Solve the 2D Poisson equation ∇²u = f using finite differences with ghost cells.

    Parameters:
    f (np.ndarray): 2d source term array for interior (nx, ny).
    dx (float): grid spacing in x-direction.
    dy (float): grid spacing in y-direction.

    Returns:
    np.ndarray: solution u ((nx+2) x (ny+2)) with ghost cells set to 0.
    """
    nx, ny = f.shape
    n_total = (nx + 2) * (ny + 2)

    # 1D Laplacian operators for full grid
    Dx = diags([1, -2, 1], [-1, 0, 1], shape=(nx + 2, nx + 2)) / dx**2
    Dy = diags([1, -2, 1], [-1, 0, 1], shape=(ny + 2, ny + 2)) / dy**2

    # Identity matrices
    Ix = identity(nx + 2)
    Iy = identity(ny + 2)

    # 2D Laplacian: A = Iy ⊗ Dx + Dy ⊗ Ix
    A = (kron(Iy, Dx) + kron(Dy, Ix)).tolil()

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
    u_flat = spsolve(A, b)

    # Reshape to 2D
    u = u_flat.reshape((nx + 2, ny + 2))

    return u


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nx, ny = 100, 100
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)  # Source term
    u = solve_poisson_2d(f, dx, dy)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    error = np.max(np.abs(u[1:-1, 1:-1] - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto")
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, u[1:-1, 1:-1], shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Numerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

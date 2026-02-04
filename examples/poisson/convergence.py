import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

file_path = os.path.dirname(os.path.realpath(__file__))
Lx, Ly = 1.0, 1.0
G = 3

n_range = 2 ** np.arange(3, 8, dtype=int)
errors_u = np.zeros(n_range.size)
errors_du = np.zeros(n_range.size)
for i, n in enumerate(n_range):
    nx = ny = n
    dx = Lx / nx
    dy = Ly / ny
    x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
    y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
    with h5py.File(
        f"{file_path}/poisson_{n}_0.h5",
        "r",
    ) as f:
        phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
        Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
        Ey = f["VTKHDF/CellData/Ey"][:].reshape(nx + 2 * G, ny + 2 * G)

    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = (X - 0.5) ** 2 + (Y - 0.5) ** 2 > 0.25**2
    u_exact = np.exp(-(X**2 + Y**2))
    u_exact[mask] = 0.0
    dudx_exact = -2.0 * X * np.exp(-(X**2 + Y**2))
    dudx_exact[mask] = 0.0
    dudy_exact = -2.0 * Y * np.exp(-(X**2 + Y**2))
    dudy_exact[mask] = 0.0

    errors_u[i] = np.linalg.norm((phi - u_exact)[G:-G, G:-G].flat, np.inf)
    errors_du[i] = np.linalg.norm(
        np.append(
            (-Ex - dudx_exact)[G:-G, G:-G].flat,
            (-Ey - dudy_exact)[G:-G, G:-G].flat,
        ),
        np.inf,
    )

# convergence table
print(f"Convergence (norm = {np.inf}):")
print(f"{'N':>5} {'Err_u':>14} {'Order':>8} {'Err_du':>14} {'Order':>8}")
print("-" * 55)

for i, n in enumerate(n_range):
    if i == 0:
        order_u = np.nan
        order_du = np.nan
    else:
        order_u = np.log(errors_u[i - 1] / errors_u[i]) / np.log(2)
        order_du = np.log(errors_du[i - 1] / errors_du[i]) / np.log(2)

    print(
        f"{n:5d} "
        f"{errors_u[i]:14.2e} "
        f"{order_u:8.2f} "
        f"{errors_du[i]:14.2e} "
        f"{order_du:8.2f}"
    )

plt.figure()
plt.subplot(121)
plt.loglog(1 / n_range, errors_u, "o-", label="actual")
plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
plt.xlabel("h")
plt.ylabel("err")
plt.legend()
plt.title("Convergence of $u$")
plt.subplot(122)
plt.loglog(1 / n_range, errors_du, "o-", label="actual")
plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
plt.xlabel("h")
plt.ylabel("err")
plt.legend()
plt.title("Convergence of $\\nabla u$")
plt.show()

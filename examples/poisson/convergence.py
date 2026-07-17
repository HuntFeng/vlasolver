import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

def surface(x, y):
    rr = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    ang = np.arctan2(y - y0, x - x0)
    return rr - (0.5 + 0.15 * np.sin(5 * ang))

file_path = os.path.dirname(os.path.realpath(__file__))
G = 3

x0 = 0.02 * np.sqrt(5)
y0 = 0.02 * np.sqrt(3)
eps_safe = 1e-30
n_range = 2 ** np.arange(3, 10, dtype=int)
errors_u = np.zeros(n_range.size)
errors_du = np.zeros(n_range.size)
for i, n in enumerate(n_range):
    nx = n 
    ny = n
    dx = 2.0 / nx
    dy = 2.0 / ny
    x = np.arange(-1.0 + dx / 2 -G*dx, 1.0 + dx / 2 + G*dx, dx)
    y = np.arange(-1.0 + dy / 2 -G*dy, 1.0 + dy / 2 + G*dy, dy)
    with h5py.File(
        f"{file_path}/poisson_{n}_0.h5",
        "r",
    ) as f:
        phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
        Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
        Ey = f["VTKHDF/CellData/Ey"][:].reshape(nx + 2 * G, ny + 2 * G)

    X, Y = np.meshgrid(x, y, indexing="ij")
    # Region mask via the level-set sign at every grid point.
    Phi = surface(X, Y)
    mask_minus = Phi < 0  # Omega^-: inside the irregular shape.

    R2 = X**2 + Y**2
    R2_safe = np.maximum(R2, eps_safe)

    # u^+ - u^- and the two-sided u/grad fields, defined on the full grid.
    u_minus = R2.copy()
    u_plus = 0.1 * R2**2 - 0.01 * np.log(2.0 * np.sqrt(R2_safe))
    u_exact = np.where(mask_minus, u_minus, u_plus)

    dudx_minus = 2.0 * X
    dudy_minus = 2.0 * Y
    coeff_plus = 0.4 * R2 - 0.01 / R2_safe
    dudx_plus = X * coeff_plus
    dudy_plus = Y * coeff_plus
    dudx_exact = np.where(mask_minus, dudx_minus, dudx_plus)
    dudy_exact = np.where(mask_minus, dudy_minus, dudy_plus)

    u = phi
    dudx, dudy = -Ex, -Ey

    errors_u[i] = np.linalg.norm((u - u_exact)[G:-G, G:-G].flat, np.inf)
    errors_du[i] = np.linalg.norm(
        np.append(
            (dudx - dudx_exact)[G:-G, G:-G].flat,
            (dudy - dudy_exact)[G:-G, G:-G].flat,
        ),
        np.inf,
    )
    print(
        f"n={n}, Max error: {errors_u[i]:.3e}, "
        f"Max grad error: {errors_du[i]:.3e}"
    )

print(f"Convergence (norm = {np.inf}):")
print(f"{'N':>5} {'Err_u':>14} {'Order':>8} {'Err_du':>14} {'Order':>8}")
print("-" * 55)
for idx, n in enumerate(n_range):
    if idx == 0:
        order_u = np.nan
        order_du = np.nan
    else:
        order_u = np.log(errors_u[idx - 1] / errors_u[idx]) / np.log(2)
        order_du = np.log(errors_du[idx - 1] / errors_du[idx]) / np.log(2)
    print(
        f"{n:5d} "
        f"{errors_u[idx]:14.2e} "
        f"{order_u:8.2f} "
        f"{errors_du[idx]:14.2e} "
        f"{order_du:8.2f}"
    )

plt.figure()
plt.loglog(1 / n_range, errors_u, "o-", label="actual")
plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
plt.xlabel("h")
plt.ylabel("err")
plt.legend()
plt.title("Convergence of $u$")
plt.savefig(f"{file_path}/convergence_poisson_solution.png")
plt.figure()
plt.loglog(1 / n_range, errors_du, "o-", label="actual")
plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
plt.xlabel("h")
plt.ylabel("err")
plt.legend()
plt.title("Convergence of $\\nabla u$")
plt.savefig(f"{file_path}/convergence_poisson_gradient.png")

fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
ax[0].plot_surface(X, Y, u_exact, edgecolor="black", cmap=cm.coolwarm)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Exact")
ax[0].view_init(elev=85, azim=10, roll=0)
ax[1].plot_surface(X, Y, phi, edgecolor="black", cmap=cm.coolwarm)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("Numerical")
ax[1].view_init(elev=85, azim=10, roll=0)
plt.savefig(f"{file_path}/poisson_solution.png", dpi=200)
plt.show()

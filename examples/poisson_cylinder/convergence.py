import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RegularGridInterpolator

file_path = os.path.dirname(os.path.realpath(__file__))
Lx, Ly = 1.0, 0.5
G = 3

_, ax_profile = plt.subplots(figsize=(6, 3))
n = 2**7
with h5py.File(
    f"{file_path}/output_{n}_0.h5",
    "r",
) as f:
    nx = 2 * n
    ny = n
    dx = Lx / nx
    dy = Ly / ny
    x_f = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
    y_f = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
    phi_exact = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    Ex_exact = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
    Ey_exact = f["VTKHDF/CellData/Ey"][:].reshape(nx + 2 * G, ny + 2 * G)
    X_f, Y_f = np.meshgrid(x_f, y_f, indexing="ij")
    mask_f = (X_f - 0.375) ** 2 + Y_f**2 < 0.125**2
    interp_phi = RegularGridInterpolator((x_f, y_f), phi_exact)
    interp_Ex = RegularGridInterpolator((x_f, y_f), Ex_exact)
    interp_Ey = RegularGridInterpolator((x_f, y_f), Ey_exact)

    Ex_0 = 0.5 * (3 * Ex_exact[G:-G, G] - Ex_exact[G:-G, G + 1])
    ax_profile.plot(x_f[G:-G], Ex_0, "k--", label=f"n={n}")

    fig, ax = plt.subplots(figsize=(6, 3))
    c = ax.contourf(X_f, Y_f, Ex_exact, cmap="jet", levels=50)
    plt.colorbar(c)
    plt.title("$E_x$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("$x/L_x$")
    ax.set_ylabel("$y/L_x$")
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(6, 3))
    c = ax.contourf(X_f, Y_f, phi_exact, cmap="jet", levels=50)
    plt.colorbar(c)
    plt.title("$\\phi$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("$x/L_x$")
    ax.set_ylabel("$y/L_x$")
    plt.tight_layout()


n_range = 2 ** np.arange(3, 7, dtype=int)
errors_u = np.zeros(n_range.size)
errors_du = np.zeros(n_range.size)
for i, n in enumerate(n_range):
    nx = 2 * n
    ny = n
    dx = Lx / nx
    dy = Ly / ny
    x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
    y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
    with h5py.File(
        f"{file_path}/output_{n}_0.h5",
        "r",
    ) as f:
        phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
        Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
        Ey = f["VTKHDF/CellData/Ey"][:].reshape(nx + 2 * G, ny + 2 * G)

    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = (X[G:-G, G:-G] - 0.375) ** 2 + Y[G:-G, G:-G] ** 2 < 0.125**2
    u_exact = interp_phi((X[G:-G, G:-G], Y[G:-G, G:-G]))
    dudx_exact = interp_Ex((X[G:-G, G:-G], Y[G:-G, G:-G]))
    dudy_exact = interp_Ey((X[G:-G, G:-G], Y[G:-G, G:-G]))

    # errors_u[i] = np.linalg.norm(
    #     phi[G:-G, G] - u_exact[:, 0],
    #     np.inf,
    # )
    # errors_du[i] = np.linalg.norm(
    #     Ex[G:-G, G] - dudx_exact[:, 0],
    #     np.inf,
    # )
    err_u = phi[G:-G, G:-G] - u_exact
    errors_u[i] = np.linalg.norm(
        0.5 * (3 * err_u[:, 0] - err_u[:, 1]),
        np.inf,
    )
    err_du = Ex[G:-G, G:-G] - dudx_exact
    errors_du[i] = np.linalg.norm(
        0.5 * (3 * err_du[:, 0] - err_du[:, 1]),
        np.inf,
    )

    Ex_0 = 0.5 * (3 * Ex[G:-G, G] - Ex[G:-G, G + 1])
    ax_profile.plot(x[G:-G], Ex_0, "o-", label=f"n={n}")
ax_profile.set_xlabel("$x/L_x$")
ax_profile.set_ylabel("$E_x$ profiles")
ax_profile.legend()

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

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RegularGridInterpolator

file_path = os.path.dirname(os.path.realpath(__file__))
Lx, Ly = 1.0, 0.5
G = 3

n = 2**7
with h5py.File(
    f"{file_path}/poisson_{n}_0.h5",
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
    # phi_exact[mask_f] = np.nan
    # Ex_exact[mask_f] = np.nan
    # Ey_exact[mask_f] = np.nan
    interp_phi = RegularGridInterpolator((x_f, y_f), phi_exact)
    interp_Ex = RegularGridInterpolator((x_f, y_f), Ex_exact)
    interp_Ey = RegularGridInterpolator((x_f, y_f), Ey_exact)

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


def restrict(
    field_f: np.ndarray, field_c: np.ndarray, X_f: np.ndarray, Y_f: np.ndarray
):
    """restrict finer solution to a coarse grid while preserving jump"""
    ratio = int(field_f.shape[0] / field_c.shape[0])
    field_r = np.zeros_like(field_c)  # restricted field
    mask = (X_f - 0.375) ** 2 + Y_f**2 > 0.125**2
    for i in range(field_r.shape[0]):
        for j in range(field_r.shape[1]):
            I, J = i * ratio, j * ratio
            # --- Determine which side the coarse cell belongs to ---
            block_mask = mask[I : I + ratio, J : J + ratio]
            majority_side = block_mask.mean() > 0.5  # True/False

            # --- Collect only fine cells on that side ---
            block_vals = field_f[I : I + ratio, J : J + ratio]
            vals = block_vals[block_mask == majority_side]

            if len(vals) == 0:
                # Fallback: use center cell
                field_r[i, j] = field_f[I + ratio // 2, J + ratio // 2]
            else:
                field_r[i, j] = vals.mean()
    return field_r


def evaluate(
    field_f: np.ndarray,
    x_f: np.ndarray,
    y_f: np.ndarray,
    X_c: np.ndarray,
    Y_c: np.ndarray,
    mask_f: np.ndarray,
) -> np.ndarray:
    """evaluate fine field at certain locations, jump aware"""
    nx_f = len(x_f)
    ny_f = len(y_f)

    field_c = np.zeros_like(X_c)

    for idx in np.ndindex(X_c.shape):

        xc = X_c[idx]
        yc = Y_c[idx]

        # --- find lower-left fine index ---
        i = np.searchsorted(x_f, xc) - 1
        j = np.searchsorted(y_f, yc) - 1

        # clamp to valid range
        i = np.clip(i, 0, nx_f - 2)
        j = np.clip(j, 0, ny_f - 2)

        # local coordinates
        x0, x1 = x_f[i], x_f[i + 1]
        y0, y1 = y_f[j], y_f[j + 1]

        tx = (xc - x0) / (x1 - x0)
        ty = (yc - y0) / (y1 - y0)

        # surrounding nodes
        nodes = [
            (i, j, (1 - tx) * (1 - ty)),
            (i + 1, j, tx * (1 - ty)),
            (i, j + 1, (1 - tx) * ty),
            (i + 1, j + 1, tx * ty),
        ]

        # determine which side evaluation point belongs to
        side = (xc - 0.375) ** 2 + yc**2 > 0.125**2  # your interface condition

        vals = []
        weights = []

        for ii, jj, w in nodes:
            if mask_f[ii, jj] == side:
                vals.append(field_f[ii, jj])
                weights.append(w)

        if len(vals) == 0:
            # fallback to nearest neighbor
            field_c[idx] = field_f[i, j]
        else:
            weights = np.array(weights)
            vals = np.array(vals)
            field_c[idx] = np.sum(weights * vals) / np.sum(weights)

    return field_c


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
        f"{file_path}/poisson_{n}_0.h5",
        "r",
    ) as f:
        phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
        Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx + 2 * G, ny + 2 * G)
        Ey = f["VTKHDF/CellData/Ey"][:].reshape(nx + 2 * G, ny + 2 * G)

    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = (X[G:-G, G:-G] - 0.375) ** 2 + Y[G:-G, G:-G] ** 2 < 0.125**2
    # u_exact = interp_phi((X[G:-G, G:-G], Y[G:-G, G:-G]))
    # dudx_exact = interp_Ex((X[G:-G, G:-G], Y[G:-G, G:-G]))
    # dudy_exact = interp_Ey((X[G:-G, G:-G], Y[G:-G, G:-G]))
    # u_exact = restrict(
    #     phi_exact[G:-G, G:-G], phi[G:-G, G:-G], X_f[G:-G, G:-G], Y_f[G:-G, G:-G]
    # )
    # dudx_exact = restrict(
    #     Ex_exact[G:-G, G:-G], Ex[G:-G, G:-G], X_f[G:-G, G:-G], Y_f[G:-G, G:-G]
    # )
    # dudy_exact = restrict(
    #     Ey_exact[G:-G, G:-G], Ey[G:-G, G:-G], X_f[G:-G, G:-G], Y_f[G:-G, G:-G]
    # )
    u_exact = evaluate(
        phi_exact[G:-G, G:-G],
        x_f[G:-G],
        y_f[G:-G],
        X[G:-G, G:-G],
        Y[G:-G, G:-G],
        mask_f,
    )
    dudx_exact = evaluate(
        Ex_exact[G:-G, G:-G],
        x_f[G:-G],
        y_f[G:-G],
        X[G:-G, G:-G],
        Y[G:-G, G:-G],
        mask_f,
    )
    dudy_exact = evaluate(
        Ey_exact[G:-G, G:-G],
        x_f[G:-G],
        y_f[G:-G],
        X[G:-G, G:-G],
        Y[G:-G, G:-G],
        mask_f,
    )

    plt.figure()
    plt.subplot(231)
    plt.pcolormesh(X[G:-G, G:-G], Y[G:-G, G:-G], Ex[G:-G, G:-G])
    plt.title("Ex")
    plt.colorbar()
    plt.subplot(232)
    plt.pcolormesh(X[G:-G, G:-G], Y[G:-G, G:-G], dudx_exact)
    plt.title("Ex exact")
    plt.colorbar()
    plt.subplot(233)
    plt.pcolormesh(X[G:-G, G:-G], Y[G:-G, G:-G], (Ex[G:-G, G:-G] - dudx_exact))
    plt.title("Ex err")
    plt.colorbar()
    plt.subplot(234)
    plt.pcolormesh(X[G:-G, G:-G], Y[G:-G, G:-G], Ey[G:-G, G:-G])
    plt.title("Ey")
    plt.colorbar()
    plt.subplot(235)
    plt.pcolormesh(X[G:-G, G:-G], Y[G:-G, G:-G], dudy_exact)
    plt.title("Ey exact")
    plt.colorbar()
    plt.subplot(236)
    plt.pcolormesh(X[G:-G, G:-G], Y[G:-G, G:-G], (Ey[G:-G, G:-G] - dudy_exact))
    plt.title("Ey err")
    plt.colorbar()
    plt.suptitle(f"{nx}x{ny}")
    u_err = (phi[G:-G, G:-G] - u_exact)[~mask]
    if np.isnan(u_err).any():
        breakpoint()
    errors_u[i] = np.linalg.norm(
        (phi[G:-G, G:-G] - u_exact)[~mask].flat,
        np.inf,
    )
    errors_du[i] = np.linalg.norm(
        # (Ex[G:-G, G:-G] - dudx_exact).flat,
        # (Ey[G:-G, G:-G] - dudy_exact).flat,
        np.append(
            (Ex[G:-G, G:-G] - dudx_exact)[~mask].flat,
            (Ey[G:-G, G:-G] - dudy_exact)[~mask].flat,
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

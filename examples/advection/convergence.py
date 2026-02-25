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
    f"{file_path}/output_{n}_10.h5",
    "r",
) as f:
    nx = 2 * n
    ny = n
    dx = Lx / nx
    dy = Ly / ny
    x_f = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
    y_f = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
    ni_exact = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    X_f, Y_f = np.meshgrid(x_f, y_f, indexing="ij")
    interp_ni = RegularGridInterpolator((x_f, y_f), ni_exact)

    ax_profile.plot(x_f[G:-G], ni_exact[G:-G, G], "k--", label=f"n={n}, y=0")
    ax_profile.plot(x_f[G:-G], ni_exact[G:-G, -G - 1], "k-.", label=f"n={n}, y=0.5")

n_range = 2 ** np.arange(3, 7, dtype=int)
errors_ni_0 = np.zeros(n_range.size)
errors_ni_05 = np.zeros(n_range.size)

for i, n in enumerate(n_range):
    nx = 2 * n
    ny = n
    dx = Lx / nx
    dy = Ly / ny
    x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
    y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
    with h5py.File(
        f"{file_path}/output_{n}_10.h5",
        "r",
    ) as f:
        ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)

    X, Y = np.meshgrid(x, y, indexing="ij")
    # y=0 (j=G)
    ni_exact_0 = interp_ni((X[G:-G, G], Y[G:-G, G]))
    errors_ni_0[i] = np.linalg.norm(ni[G:-G, G] - ni_exact_0, np.inf)
    ax_profile.plot(x[G:-G], ni[G:-G, G], "o-", label=f"n={n}, y=0")

    # y=0.5 (j=-G-1)
    ni_exact_05 = interp_ni((X[G:-G, -G - 1], Y[G:-G, -G - 1]))
    errors_ni_05[i] = np.linalg.norm(ni[G:-G, -G - 1] - ni_exact_05, np.inf)
    ax_profile.plot(x[G:-G], ni[G:-G, -G - 1], "s-", label=f"n={n}, y=0.5")

ax_profile.set_xlabel("$x/L_x$")
ax_profile.set_ylabel("$n_i$ profiles")
ax_profile.legend()

# convergence table
print(f"Convergence at y=0 (norm = {np.inf}):")
print(f"{'N':>5} {'Err_ni':>14} {'Order':>8}")
print("-" * 32)
for i, n in enumerate(n_range):
    if i == 0:
        order_ni = np.nan
    else:
        order_ni = np.log(errors_ni_0[i - 1] / errors_ni_0[i]) / np.log(2)
    print(f"{n:5d} " f"{errors_ni_0[i]:14.2e} {order_ni:8.2f}")

print(f"\nConvergence at y=0.5 (norm = {np.inf}):")
print(f"{'N':>5} {'Err_ni':>14} {'Order':>8}")
print("-" * 32)
for i, n in enumerate(n_range):
    if i == 0:
        order_ni = np.nan
    else:
        order_ni = np.log(errors_ni_05[i - 1] / errors_ni_05[i]) / np.log(2)
    print(f"{n:5d} " f"{errors_ni_05[i]:14.2e} {order_ni:8.2f}")

plt.figure()
plt.loglog(1 / n_range, errors_ni_0, "o-", label="ni, y=0")
plt.loglog(1 / n_range, errors_ni_05, "s-", label="ni, y=0.5")
plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
plt.xlabel("h")
plt.ylabel("err")
plt.legend()
plt.title("Convergence of $n_i$")

plt.show()

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

file_path = os.path.dirname(os.path.realpath(__file__))
Lx, Ly = 1.0, 0.5
G = 3

_, ax_profile = plt.subplots(2, 1, figsize=(6, 3))
# n = 2**8
# with h5py.File(
#     f"{file_path}/output_{n}_568.h5",
#     "r",
# ) as f:
#     nx = 2 * n
#     ny = n
#     dx = Lx / nx
#     dy = Ly / ny
#     x_f = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
#     y_f = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
#     ni_exact = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
#     X_f, Y_f = np.meshgrid(x_f, y_f, indexing="ij")
#     # ni_exact = np.exp(-(((X_f - 0.2 - 0.15) / 0.1) ** 2))
#     # ni_exact[(X_f - 0.375) ** 2 + Y_f**2 <= 0.125**2] = 0
#     # ni_exact[(X_f > 0.375) & (Y_f <= 0.125)] = 0
#     # interp_ni = RegularGridInterpolator((x_f, y_f), ni_exact)
#
#     ax_profile[0].plot(x_f[G:-G], ni_exact[G:-G, G], "k--", label=f"n={n}, y=0")
#     ax_profile[1].plot(
#         x_f[G:-G], ni_exact[G:-G, ny // 2 + G], "k-.", label=f"n={n}, y=0.25"
#     )


outputs = list(filter(lambda name: name.startswith("output"), os.listdir(file_path)))
outputs = sorted(outputs, key=lambda name: int(name.split("_")[1]))
n_range = 2 ** np.arange(3, 8, dtype=int)
errors_ni_0 = np.zeros(n_range.size)
errors_ni_mid = np.zeros(n_range.size)

for i, n in enumerate(n_range):
    nx = 2 * n
    ny = n
    dx = Lx / nx
    dy = Ly / ny
    x = np.arange(dx / 2 - G * dx, Lx + G * dx, dx)
    y = np.arange(dy / 2 - G * dy, Ly + G * dy, dy)
    with h5py.File(
        # f"{file_path}/output_{n}.h5",
        f"{file_path}/{outputs[i]}",
        "r",
    ) as f:
        ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)

    X, Y = np.meshgrid(x, y, indexing="ij")
    # y=0 (j=G)
    dvx, dvy = 1.0 / 10, 1.0 / 1
    total_step = int(outputs[i].split("_")[2].split(".")[0])
    CFL = 0.9
    dt = CFL * dx / 1.0
    ni_exact = np.exp(-(((X - 0.2 - 0.15 * total_step * dt) / 0.1) ** 2))
    ni_exact[(X - 0.375) ** 2 + Y**2 <= 0.125**2] = 0
    ni_exact[(X > 0.375) & (Y <= 0.125)] = 0

    norm = np.inf
    # ni_exact_0 = interp_ni((X[G:-G, G], Y[G:-G, G]))
    ni_exact_0 = ni_exact[G:-G, G]
    errors_ni_0[i] = np.linalg.norm(ni[G:-G, G] - ni_exact_0, norm)
    ax_profile[0].plot(x[G:-G], ni[G:-G, G], "o-", label=f"n={n}, y=0")

    # y=0.25 (j=ny//2)
    # ni_exact_mid = interp_ni((X[G:-G, (ny + 2 * G) // 2], Y[G:-G, (ny + 2 * G) // 2]))
    ni_exact_mid = ni_exact[G:-G, ny // 2 + G]
    errors_ni_mid[i] = np.linalg.norm(ni[G:-G, ny // 2 + G] - ni_exact_mid, norm)
    ax_profile[1].plot(x[G:-G], ni[G:-G, ny // 2 + G], "s-", label=f"n={n}, y=0.25")

ax_profile[1].set_xlabel("$x/L_x$")
ax_profile[0].set_ylabel("$n_i$ profiles")
ax_profile[1].set_ylabel("$n_i$ profiles")
ax_profile[0].legend()
ax_profile[1].legend()

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

print(f"\nConvergence at y=0.25 (norm = {np.inf}):")
print(f"{'N':>5} {'Err_ni':>14} {'Order':>8}")
print("-" * 32)
for i, n in enumerate(n_range):
    if i == 0:
        order_ni = np.nan
    else:
        order_ni = np.log(errors_ni_mid[i - 1] / errors_ni_mid[i]) / np.log(2)
    print(f"{n:5d} " f"{errors_ni_mid[i]:14.2e} {order_ni:8.2f}")

plt.figure()
plt.loglog(1 / n_range, errors_ni_0, "o-", label="ni, y=0")
plt.loglog(1 / n_range, errors_ni_mid, "s-", label="ni, y=0.25")
plt.loglog(
    1 / n_range, errors_ni_mid[0] * n_range[0] ** 3 / n_range**3, "--", label="$O(h^3)$"
)
plt.loglog(
    1 / n_range, errors_ni_0[0] * n_range[0] ** 2 / n_range**2, "--", label="$O(h^2)$"
)
# plt.loglog(
#     1 / n_range, errors_ni_0[0] * n_range[0] ** 1 / n_range**1, "--", label="$O(h^1)$"
# )
plt.xlabel("h")
plt.ylabel("err")
plt.legend()
plt.title("Convergence of $n_i$")

plt.show()

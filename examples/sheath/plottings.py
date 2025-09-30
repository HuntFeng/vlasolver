import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

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

nx, ny, nvx, nvy = 10, 125, 30, 110
x_min, y_min = 0, 0
Lx, Ly = 1.0, 1.0  # normalized to 1

vx_min_e, vy_min_e = -4, -5
Lvx_e, Lvy_e = 8, 10
# in simulation, the ion velocity ranges are multiplied by vr
vx_min_i, vy_min_i = -4, -15
Lvx_i, Lvy_i = 8, 16
G = 3
step = 8000
is_include_ghost = True
Te = 1.0  # eV
Ti = 0.1  # eV
me = 1.0
mi = 2 * 1836.0
mr = mi / me
Tr = Ti / Te
vr = np.sqrt(Tr / mr)
u0 = np.sqrt(Te / mi)


file_path = os.path.dirname(os.path.realpath(__file__))
with h5py.File(
    f"{file_path}/../../data/sheath/output_{step:04d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)
    fi = f["VTKHDF/CellData/fi"][:].reshape(
        nx + 2 * G, ny + 2 * G, nvx + 2 * G, nvy + 2 * G
    )
    fe = f["VTKHDF/CellData/fe"][:].reshape(
        nx + 2 * G, ny + 2 * G, nvx + 2 * G, nvy + 2 * G
    )

if is_include_ghost:
    dx, dy = Lx / nx, Ly / ny
    x = np.arange(x_min - G * dx + dx / 2, x_min + Lx + G * dx, dx)
    y = np.arange(y_min - G * dy + dy / 2, y_min + Ly + G * dy, dy)
    dvx_e, dvy_e = Lvx_e / nvx, Lvy_e / nvy
    vx_e = np.arange(
        vx_min_e - G * dvx_e + dvx_e / 2, vx_min_e + Lvx_e + G * dvx_e, dvx_e
    )
    vy_e = np.arange(
        vy_min_e - G * dvy_e + dvy_e / 2, vy_min_e + Lvy_e + G * dvy_e, dvy_e
    )
    dvx_i, dvy_i = Lvx_i / nvx, Lvy_i / nvy
    vx_i = np.arange(
        vx_min_i - G * dvx_i + dvx_i / 2, vx_min_i + Lvx_i + G * dvx_i, dvx_i
    )
    vy_i = np.arange(
        vy_min_i - G * dvy_i + dvy_i / 2, vy_min_i + Lvy_i + G * dvy_i, dvy_i
    )
else:
    ni = ni[G:-G, G:-G]
    ne = ne[G:-G, G:-G]
    phi = phi[G:-G, G:-G]
    fi = fi[G:-G, G:-G, G:-G, G:-G]
    fe = fe[G:-G, G:-G, G:-G, G:-G]

    dx, dy = Lx / nx, Ly / ny
    x = np.arange(x_min + dx / 2, x_min + Lx, dx)
    y = np.arange(y_min + dy / 2, y_min + Ly, dy)

    dvx_e, dvy_e = Lvx_e / nvx, Lvy_e / nvy
    vx_e = np.arange(vx_min_e + dvx_e / 2, vx_min_e + Lvx_e, dvx_e)
    vy_e = np.arange(vy_min_e + dvy_e / 2, vy_min_e + Lvy_e, dvy_e)
    dvx_i, dvy_i = Lvx_i / nvx, Lvy_i / nvy
    vx_i = np.arange(vx_min_i + dvx_i / 2, vx_min_i + Lvx_i, dvx_i)
    vy_i = np.arange(vy_min_i + dvy_i / 2, vy_min_i + Lvy_i, dvy_i)


phi_w = -np.log(np.sqrt(mr / (2 * np.pi)))
n_ea = np.zeros(ne.shape[1])
n_ia = np.zeros(ne.shape[1])
dvy_i = dvy_i * vr
vy_i = vy_i * vr
f_ea = np.zeros((fi.shape[1], fi.shape[3]))
f_ia = np.zeros((fi.shape[1], fi.shape[3]))
phi_a = np.tile(np.loadtxt(f"{file_path}/initial_potential.csv"), (phi.shape[0], 1))
# phi_a = phi.copy()
for j in range(ne.shape[1]):
    v_ce = np.sqrt(2 * (phi_a[nx // 2, j] - phi_w))
    for jv, vy in enumerate(vy_e):
        if vy <= v_ce:
            f_ea[j, jv] = np.exp(-(vy**2) / 2 + phi_a[nx // 2, j]) / np.sqrt(2 * np.pi)
            n_ea[j] += f_ea[j, jv] * dvy_e

    v_ci = -np.sqrt(2 * np.abs(phi_a[nx // 2, j]) / mr)
    for jv, vy in enumerate(vy_i):
        if vy <= v_ci:
            f_ia[j, jv] = (
                np.exp(-((np.sqrt(vy**2 - v_ci**2) - u0) ** 2) / (2 * vr**2))
                / np.sqrt(2 * np.pi)
                / vr
            )
            n_ia[j] += f_ia[j, jv] * dvy_i

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
phi_norm = phi[phi.shape[0] // 2, :] / (2 * Tr)
plt.plot(y, phi_norm, label="$\\phi$")
plt.plot(
    y[G:-G:3],
    phi_a[phi_a.shape[0] // 2, G:-G:3] / (2 * Tr),
    "o",
    alpha=0.5,
    label="$\\phi_a$",
)
plt.legend()
plt.ylabel("$e\\phi/2k_BT_i$")
plt.xlabel("$y/L_y$")
plt.subplot(1, 2, 2)
plt.plot(y, ne[ne.shape[0] // 2, :], label="$n_e$")
plt.plot(y, ni[ni.shape[0] // 2, :], label="$n_i$")
plt.plot(y[G:-G:3], n_ea[G:-G:3], "o", alpha=0.5, label="$n_{ea}$")
plt.plot(y[G:-G:3], n_ia[G:-G:3], "^", alpha=0.5, label="$n_{ia}$")
plt.legend()
plt.xlabel("$y/L_y$")
plt.ylabel("$n/n_0$")
plt.tight_layout()
plt.savefig(f"{file_path}/potential_and_density.png")

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
VY_e, Y = np.meshgrid(vy_e, y)
plt.contourf(
    Y,
    VY_e,
    fe.sum(axis=2)[fe.shape[0] // 2, :, :] * dvx_e,
    cmap="jet",
    levels=50,
    vmin=0,
)
plt.yticks(np.arange(-5, 6, 2))
plt.ylabel("$v_y/v_{th,e}$")
plt.xlabel("$y/L_y$")
plt.title("$f_e$")
plt.subplot(2, 2, 3)
plt.contourf(
    Y,
    VY_e,
    f_ea,
    cmap="jet",
    levels=50,
    vmin=0,
)
plt.yticks(np.arange(-5, 6, 2))
plt.ylabel("$v_y/v_{th,e}$")
plt.xlabel("$y/L_y$")
plt.title("$f_{ea}$")
plt.subplot(2, 2, 2)
VY_i, Y = np.meshgrid(vy_i / vr, y)
plt.contourf(
    Y,
    VY_i,
    fi[fi.shape[0] // 2, :, fi.shape[2] // 2, :],
    cmap="jet",
    levels=50,
    vmin=0,
)
plt.ylim(vy_min_i, vy_min_i + Lvy_i)
plt.xlabel("$y/L_y$")
plt.ylabel("$v_y/v_{th,i}$")
plt.title("$f_i$")

plt.subplot(2, 2, 4)
plt.contourf(
    Y,
    VY_i,
    f_ia,
    cmap="jet",
    levels=50,
    vmin=0,
)
plt.ylim(vy_min_i, vy_min_i + Lvy_i)
plt.xlabel("$y/L_y$")
plt.ylabel("$v_y/v_{th,i}$")
plt.title("$f_{ia}$")

plt.tight_layout()
plt.savefig(f"{file_path}/distribution.png")


plt.show()

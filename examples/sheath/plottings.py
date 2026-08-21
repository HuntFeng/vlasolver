import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

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
step = 3000
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

# Always include ghost cells
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

phi_w = -np.log(np.sqrt(mr / (2 * np.pi)))
n_ea = np.zeros(ne.shape[1])
n_ia = np.zeros(ne.shape[1])
dvy_i = dvy_i * vr
vy_i = vy_i * vr
f_ea = np.zeros((fi.shape[1], fi.shape[3]))
f_ia = np.zeros((fi.shape[1], fi.shape[3]))
phi_a = phi.copy()
cx = phi.shape[0] // 2
for j in range(ne.shape[1]):
    v_ce = np.sqrt(2 * (phi_a[cx, j] - phi_w))
    for jv, vy_val in enumerate(vy_e):
        if vy_val <= v_ce:
            f_ea[j, jv] = np.exp(-(vy_val**2) / 2 + phi_a[cx, j]) / np.sqrt(2 * np.pi)

    v_ci = -np.sqrt(2 * np.abs(phi_a[cx, j]) / mr)
    for jv, vy_val in enumerate(vy_i):
        if vy_val <= v_ci:
            f_ia[j, jv] = (
                np.exp(-((np.sqrt(vy_val**2 - v_ci**2) - u0) ** 2) / (2 * vr**2))
                / np.sqrt(2 * np.pi)
                / vr
            )

n_ea = np.exp(phi_a.mean(axis=0))
n_ia = 1.0 / np.sqrt(1 - 2 * phi_a.mean(axis=0))

# === 1D potential and density ===
theoretical_potential = -np.log(np.sqrt(mi / (2 * np.pi * me))) / (2 * Tr)
wall_potential = phi[cx, G] / (2 * Tr)
print(
    f"theoretical wall potential {theoretical_potential}"
)
print(f"simulation wall potential {wall_potential}")

# === Analytical sheath potential ===
# Collisionless Bohm sheath solved as a boundary-value problem in normalized
# units: phi in Te/e, lengths in Debye lengths, n0 = 1. Boltzmann electrons
# (n_e = exp(phi)) and cold Bohm ions (n_i = 1 / sqrt(1 - 2*phi)) give Poisson's
# equation
#     d^2 phi / dy^2 = n_e - n_i = exp(phi) - 1/sqrt(1 - 2*phi),
# with phi = phi_w at the wall and phi = 0 in the bulk.
Ly_debye = 20.0  # domain length in Debye lengths (input.ini: Ly = 20)
y_wall = 0.0  # wall surface where the sheath begins (normalized to Ly)
phi_w = -np.log(np.sqrt(mi / (2 * np.pi * me)))  # wall potential in Te/e
L_sheath = (1 - y_wall) * Ly_debye  # wall-to-bulk distance in Debye lengths


def sheath_ode(s, y):
    phi = y[0]
    ne = np.exp(phi)  # Boltzmann electrons
    arg = np.maximum(1.0 - 2.0 * phi, 1e-12)
    ni = 1.0 / np.sqrt(arg)  # cold Bohm ions
    d2phi = ne - ni  # normalized Poisson (Debye lengths, Te/e)
    return np.vstack((y[1], d2phi))


def bc(ya, yb):
    return np.array([
        ya[0] - phi_w,  # wall potential
        yb[0],  # bulk (plasma) potential = 0
    ])


s_mesh = np.linspace(0.0, L_sheath, 400)
phi_guess = phi_w * np.exp(-s_mesh / (L_sheath / 10))
y_guess = np.vstack((phi_guess, np.gradient(phi_guess, s_mesh)))

sol = solve_bvp(sheath_ode, bc, s_mesh, y_guess, max_nodes=10000)
if not sol.success:
    print(sol.message)

s_plot = np.linspace(0.0, L_sheath, 1000)
y_analytic = y_wall + s_plot / Ly_debye
phi_bvp = sol.sol(s_plot)[0]  # analytical potential in Te/e
phi_analytic = phi_bvp / (2 * Tr)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
phi_norm = phi.mean(axis=0) / (2 * Tr)
plt.plot(y, phi_norm, label="$\\phi$")
plt.plot(y_analytic, phi_analytic, "-", color="red", label="$\\phi$ (analytic)")
plt.ylim(wall_potential*1.1, 1)
plt.xlim(0, 1)
plt.legend()
plt.ylabel("$e\\phi/2k_BT_i$")
plt.xlabel("$y/L_y$")
plt.subplot(1, 2, 2)
plt.plot(y, ne.mean(axis=0), label="$n_e$")
plt.plot(y, ni.mean(axis=0), label="$n_i$")
plt.plot(y[G:-G:3], n_ea[G:-G:3], "o", alpha=0.5, label="$n_{ea}$")
plt.plot(y[G:-G:3], n_ia[G:-G:3], "^", alpha=0.5, label="$n_{ia}$")
plt.xlim(0, 1)
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
plt.xlim(0, Ly)
plt.ylim(vy_min_e, vy_min_e + Lvy_e)

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
plt.xlim(0, Ly)
plt.ylim(vy_min_e, vy_min_e + Lvy_e)

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
plt.xlim(0, Ly)

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
plt.xlim(0, Ly)

plt.tight_layout()
plt.savefig(f"{file_path}/distribution.png")

X, Y = np.meshgrid(x, y, indexing="ij")

plt.figure()
plt.contourf(
    X,
    Y,
    ne,
    cmap="jet",
    levels=20,
    vmin=0,
)
plt.colorbar()
plt.contour(X, Y, ne, levels=20, colors="black", linestyles="solid")
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$n_e$")
plt.savefig(f"{file_path}/number_density_electron.png")

plt.figure()
plt.contourf(
    X,
    Y,
    ni,
    cmap="jet",
    levels=20,
    vmin=0,
)
plt.colorbar()
plt.contour(X, Y, ni, levels=20, colors="black", linestyles="solid")
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$n_i$")
plt.savefig(f"{file_path}/number_density_ion.png")

plt.figure()
plt.contourf(
    X,
    Y,
    ni - ne,
    cmap="jet",
    levels=20,
    vmin=0,
)
plt.colorbar()
plt.contour(X, Y, ni - ne, levels=20, colors="black", linestyles="solid")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$\\rho$")
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.savefig(f"{file_path}/charge_density.png")

plt.figure()
plt.contourf(X, Y, phi / (2 * Tr), levels=20, cmap="jet")
plt.colorbar()
plt.contour(X, Y, phi / (2 * Tr), levels=20, colors="black", linestyles="solid")
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$e\\phi/2k_BT_i$")
plt.savefig(f"{file_path}/potential.png")

plt.show()

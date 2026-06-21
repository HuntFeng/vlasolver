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

nx, ny = 10, 125
x_min, y_min = 0, 0
Lx, Ly = 1.0, 1.0  # normalized to 1
G = 3
# step = 20500
step = 18500
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
    f"{file_path}/../../data/sheath_oml/output_{step:05d}.h5",
    "r",
) as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx + 2 * G, ny + 2 * G)
    ne = f["VTKHDF/CellData/ne"][:].reshape(nx + 2 * G, ny + 2 * G)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx + 2 * G, ny + 2 * G)

# Always include ghost cells
dx, dy = Lx / nx, Ly / ny
x = np.arange(x_min - G * dx + dx / 2, x_min + Lx + G * dx, dx)
y = np.arange(y_min - G * dy + dy / 2, y_min + Ly + G * dy, dy)

# === 1D potential and density ===
theoretical_potential = -np.log(np.sqrt(mi / (2 * np.pi * me))) / (2 * Tr)
wall_potential = phi[phi.shape[0] // 2, G] / (2 * Tr)
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
y_wall = 2.5 / 20  # wall surface where the sheath begins (normalized to Ly)
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
phi_analytic = sol.sol(s_plot)[0] / (2 * Tr)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
phi_norm = phi.mean(axis=0) / (2 * Tr)
plt.plot(y, phi_norm, "o", label="$\\phi$ (sim)")
plt.plot(y_analytic, phi_analytic, "-", color="red", label="$\\phi$ (analytic)")
plt.ylim(wall_potential*1.1, 1)
plt.axvline(2.5 / 20, color="black", linestyle="--")
plt.xlim(0, 1)
plt.legend()
plt.ylabel("$e\\phi/2k_BT_i$")
plt.xlabel("$y/L_y$")
plt.subplot(1, 2, 2)
plt.plot(y, ne.mean(axis=0), "o", label="$n_e$")
plt.plot(y, ni.mean(axis=0), "o", label="$n_i$")
plt.axvline(2.5 / 20, color="black", linestyle="--")
plt.xlim(0, 1)
plt.legend()
plt.xlabel("$y/L_y$")
plt.ylabel("$n/n_0$")
plt.tight_layout()
plt.savefig(f"{file_path}/potential_and_density.png")

plt.figure()
Ey = (np.roll(phi_norm, -1) - np.roll(phi_norm, 1)) / (2*dy)
plt.plot(y, Ey, "o")
plt.axvline(2.5 / 20, color="black", linestyle="--")

plt.show()

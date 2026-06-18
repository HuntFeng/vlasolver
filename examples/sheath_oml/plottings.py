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

nx, ny = 10, 125
x_min, y_min = 0, 0
Lx, Ly = 1.0, 1.0  # normalized to 1
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
    f"{file_path}/../../data/sheath_oml/output_{step:04d}.h5",
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
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
phi_norm = phi.mean(axis=0) / (2 * Tr)
plt.plot(y, phi_norm, label="$\\phi$")
plt.ylim(wall_potential*1.1, 1)
plt.axvline(2.5 / 20, color="black", linestyle="--")
plt.xlim(0, 1)
plt.legend()
plt.ylabel("$e\\phi/2k_BT_i$")
plt.xlabel("$y/L_y$")
plt.subplot(1, 2, 2)
plt.plot(y, ne.mean(axis=0), label="$n_e$")
plt.plot(y, ni.mean(axis=0), label="$n_i$")
plt.axvline(2.5 / 20, color="black", linestyle="--")
plt.xlim(0, 1)
plt.legend()
plt.xlabel("$y/L_y$")
plt.ylabel("$n/n_0$")
plt.tight_layout()
plt.savefig(f"{file_path}/potential_and_density.png")


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

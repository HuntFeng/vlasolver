import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from IPython.display import HTML

nx, ny, nvx, nvy = 134, 70, 134, 134
Lx, Ly = 1, 0.5
ngc = 3
step = 20
# with h5py.File(f"data/debug_potential/output_{step:01d}.h5", "r") as f:
with h5py.File(f"data/debug_vlasov/output_{step:03d}.h5", "r") as f:
    # with h5py.File(f"data/202507101450/output_{step:03d}.h5", "r") as f:
    # with h5py.File(f"data/plasma_past_charged_cylinder/output_{step:03d}.h5", "r") as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx, ny)
    Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx, ny)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx, ny)

x = np.linspace(0, Lx, nx - 2 * ngc)
y = np.linspace(0, Ly, ny - 2 * ngc)
vx = np.linspace(-10, 10, nvx - 2 * ngc)
vy = np.linspace(-10, 10, nvy - 2 * ngc)
Y, X = np.meshgrid(y, x)

n = ni[ngc:-ngc, ngc:-ngc] / ni[ngc:-ngc, ngc:-ngc].max()
# n[(X - 0.375) ** 2 + Y**2 <= 0.125**2] = np.nan
ind_neg = np.argwhere(n < 0.0)
if ind_neg.shape[0] > 0:
    print("negative n occurs at", ind_neg)
    negativity = np.zeros_like(n) * np.nan
    negativity[ind_neg[:, 0], ind_neg[:, 1]] = -100
    negativity[(np.abs(X - 0.2) <= 0.02) & (np.abs(Y - 0.1) <= 0.1)] = 1
    plt.figure(figsize=(6, 3))
    plt.pcolormesh(X, Y, negativity, cmap="jet")
    plt.colorbar()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Negative Ion Density")
plt.figure(figsize=(6, 3))
plt.pcolormesh(X, Y, n, cmap="jet")
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Ion Number Density")
#
# plt.figure(figsize=(6, 3))
# phi0 = 0.3
# Te = 1.5
# ne = np.exp((phi - phi0) / Te)
# # n = ne[ngc:-ngc, ngc:-ngc] / ne[ngc:-ngc, ngc:-ngc].max()
# n = ne[ngc:-ngc, ngc:-ngc]
# plt.pcolormesh(X, Y, n, cmap="jet")
# plt.colorbar()
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.title("Electron Number Density")

plt.figure(figsize=(6, 3))
plt.pcolormesh(X, Y, Ex[ngc:-ngc, ngc:-ngc], cmap="jet")
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$E_x$")

plt.figure(figsize=(6, 3))
# _phi = phi.copy()[ngc:-ngc, ngc:-ngc]
# _phi[(X - 0.375) ** 2 + Y**2 <= 0.125**2] = np.nan
_phi = phi[ngc:-ngc, ngc:-ngc]
plt.pcolormesh(X, Y, _phi, cmap="jet")
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Potential")


plt.figure()
plt.plot(x, Ex[ngc:-ngc, ngc], label="$y=0$")
plt.plot(x, Ex[ngc:-ngc, -ngc - 1], label="$y=0.5$")
plt.xlabel("$x$")
plt.ylabel("$E_x$")
plt.legend()

plt.figure()
plt.plot(phi[:, ngc], label="$y=0$")
plt.plot(phi[:, -ngc], label="$y=0.5$")
plt.xlabel("$x$")
plt.ylabel("$\\phi$")
plt.legend()
plt.show()

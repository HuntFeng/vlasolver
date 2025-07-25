import h5py
import matplotlib.pyplot as plt
import numpy as np

nx, ny = 134, 70
Lx, Ly = 1, 0.5
ngc = 3
step = 500
# with h5py.File(f"data/debug_vlasov/output_{step:03d}.h5", "r") as f:
with h5py.File(f"data/plasma_past_charged_cylinder/output_{step:03d}.h5", "r") as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx, ny)
    Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx, ny)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx, ny)

dx = Lx / (nx - 2 * ngc)
dy = Ly / (ny - 2 * ngc)
x = np.arange(-2.5 * dx, Lx + 3 * dx, dx)
y = np.arange(-2.5 * dy, Ly + 3 * dy, dy)
Y, X = np.meshgrid(y, x)


n = ni / ni.max()
ind_neg = np.argwhere(n < 0.0)
if ind_neg.shape[0] > 0:
    print("negative n occurs at", ind_neg)
    negativity = np.zeros_like(n) * np.nan
    negativity[ind_neg[:, 0], ind_neg[:, 1]] = -100
    # negativity[(np.abs(X - 0.15) <= 0.04) & (np.abs(Y - 0.1) <= 0.1)] = 1
    n[(X - 0.375) ** 2 + Y**2 <= 0.125**2] = np.nan
    plt.figure(figsize=(6, 3))
    plt.imshow(negativity, cmap="jet")
    plt.colorbar()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Negative Ion Density")
plt.figure(figsize=(6, 3))
# n[(np.abs(X - 0.15) <= 0.04) & (np.abs(Y - 0.1) <= 0.1)] = np.nan
n[(X - 0.375) ** 2 + Y**2 <= 0.125**2] = np.nan
plt.pcolormesh(X, Y, n, cmap="jet")
# plt.imshow(n.T, cmap="jet", origin="lower", interpolation="bilinear")
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Ion Number Density")

plt.figure(figsize=(6, 3))
# n[(np.abs(X - 0.15) <= 0.04) & (np.abs(Y - 0.1) <= 0.1)] = np.nan
ne = np.exp((phi - 0.3) / 1.5)
ne = ne * np.nanmax(ni)
ne[(X - 0.375) ** 2 + Y**2 <= 0.125**2] = np.nan
plt.pcolormesh(X, Y, ne, cmap="jet")
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Electron Number Density")

plt.figure(figsize=(6, 3))
plt.pcolormesh(X, Y, Ex, cmap="jet")
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$E_x$")

plt.figure(figsize=(6, 3))
plt.pcolormesh(X, Y, phi, cmap="jet")
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Potential")

plt.figure()
plt.plot(x, Ex[:, ngc], label="$y=0$")
plt.plot(x, Ex[:, -ngc - 1], label="$y=0.5$")
plt.xlabel("$x$")
plt.ylabel("$E_x$")
plt.legend()

plt.figure()
plt.plot(x, phi[:, ngc], label="$y=0$")
plt.plot(x, phi[:, -ngc], label="$y=0.5$")
plt.xlabel("$x$")
plt.ylabel("$\\phi$")
plt.legend()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# negative_f = np.zeros_like(ni) * np.nan
# negative_f[14:17, :] = -1
# negative_f[(np.abs(X - 0.15) <= 0.04) & (np.abs(Y - 0.1) <= 0.1)] = 1
# ax.imshow(negative_f.T, cmap="jet", origin="lower")
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
plt.show()

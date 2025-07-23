import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

nx, ny, nvx, nvy = 134, 70, 134, 134
Lx, Ly = 1, 0.5
ngc = 3
step = 1
with h5py.File(f"data/debug_vlasov/output_{step:d}.h5", "r") as f:
    ni = f["VTKHDF/CellData/ni"][:].reshape(nx, ny)
    Ex = f["VTKHDF/CellData/Ex"][:].reshape(nx, ny)
    phi = f["VTKHDF/CellData/phi"][:].reshape(nx, ny)

dx = Lx / (nx - 2 * ngc)
dy = Ly / (ny - 2 * ngc)
x = np.arange(-2.5 * dx, Lx + 3 * dx, dx)
y = np.arange(-2.5 * dy, Ly + 3 * dy, dy)
Y, X = np.meshgrid(y, x)

# n = ni / ni.max()
# print("n.shape", n.shape, "X.shape", X.shape, "Y.shape", Y.shape)
# ind_neg = np.argwhere(n < 0.0)
# if ind_neg.shape[0] > 0:
#     print("negative n occurs at", ind_neg)
#     negativity = np.zeros_like(n) * np.nan
#     negativity[ind_neg[:, 0], ind_neg[:, 1]] = -100
#     negativity[(np.abs(X - 0.15) <= 0.04) & (np.abs(Y - 0.1) <= 0.1)] = 1
#     plt.figure(figsize=(6, 3))
#     plt.imshow(negativity, cmap="jet")
#     plt.colorbar()
#     plt.xlabel("$x$")
#     plt.ylabel("$y$")
#     plt.title("Negative Ion Density")
# plt.figure(figsize=(6, 3))
# plt.pcolormesh(X, Y, n, cmap="jet")
# plt.colorbar()
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.title("Ion Number Density")
# #
# # plt.figure(figsize=(6, 3))
# # phi0 = 0.3
# # Te = 1.5
# # ne = np.exp((phi - phi0) / Te)
# # # n = ne[ngc:-ngc, ngc:-ngc] / ne[ngc:-ngc, ngc:-ngc].max()
# # n = ne[ngc:-ngc, ngc:-ngc]
# # plt.pcolormesh(X, Y, n, cmap="jet")
# # plt.colorbar()
# # plt.xlabel("$x$")
# # plt.ylabel("$y$")
# # plt.title("Electron Number Density")
#
# plt.figure(figsize=(6, 3))
# plt.pcolormesh(X, Y, Ex, cmap="jet")
# plt.colorbar()
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.title("$E_x$")
#
# plt.figure(figsize=(6, 3))
# plt.pcolormesh(X, Y, phi, cmap="jet")
# plt.colorbar()
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.title("Potential")
#
# plt.figure()
# plt.plot(x, Ex[:, ngc], label="$y=0$")
# plt.plot(x, Ex[:, -ngc - 1], label="$y=0.5$")
# plt.xlabel("$x$")
# plt.ylabel("$E_x$")
# plt.legend()
#
# plt.figure()
# plt.plot(x, phi[:, ngc], label="$y=0$")
# plt.plot(x, phi[:, -ngc], label="$y=0.5$")
# plt.xlabel("$x$")
# plt.ylabel("$\\phi$")
# plt.legend()

fig = plt.figure()
ax = fig.add_subplot(111)
negative_f = np.zeros_like(ni) * np.nan
negative_f[3, 40] = -1
negative_f[(np.abs(X - 0.15) <= 0.04) & (np.abs(Y - 0.1) <= 0.1)] = 1
ax.set_xlabel("$y$")
ax.set_ylabel("$x$")
plt.show()

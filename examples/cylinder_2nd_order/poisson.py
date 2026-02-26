import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

Lx = 1.0
Ly = 0.5
nx_int = 160
ny_int = 50
ngc = 1
dx = Lx / nx_int
dy = Ly / ny_int
nx = nx_int + 2 * ngc
ny = ny_int + 2 * ngc
x_arr = np.arange((0.5 - ngc) * dx, Lx + ngc * dx, dx)
y_arr = np.arange((0.5 - ngc) * dy, Ly + ngc * dy, dy)

surface = lambda x, y: (x - 0.375) ** 2 + y**2 - 0.125**2  # level set
normal = lambda x, y: (
    (x - 0.375) / np.sqrt((x - 0.375) ** 2 + y**2),
    y / np.sqrt((x - 0.375) ** 2 + y**2),
)
normal = lambda x, y: (-normal(x, y)[1], normal(x, y)[0])
phi = np.zeros((nx, ny))  # potential
a = np.zeros((nx, ny))  # normal jump of [u]
a_tau = np.zeros((nx, ny))  # tangential jump of [u], to be defined at each grid cell
b = np.zeros((nx, ny))  # normal [beta*u_n]

A = np.zeros((nx * ny, nx * ny))  # sparse matrix to be solved
for i in range(nx):
    for j in range(ny):
        x, y = x_arr[i], y_arr[j]
        I = i * ny + j  # at I-th row of A
        # normal cell
        A[I, (i + 1) * ny + j] = 1 / dx**2
        A[I, (i - 1) * ny + j] = 1 / dx**2
        A[I, i * ny + j] = -2 / dx**2

        # near boundary
        eta = surface(x, y)
        eta_l = surface(x - dx, y)
        eta_r = surface(x + dx, y)
        eta_b = surface(x, y - dy)
        eta_t = surface(x, y + dy)

        n1, n2 = normal(x, y)

        theta_l = 1
        theta_r = 1
        theta_b = 1
        theta_t = 1
        if eta * eta_l < 0:
            if eta < 0:
                ...
            else:
                ...
        if eta * eta_r < 0:
            if eta < 0:
                dxx_eta = (eta_r + eta_l - 2 * eta) / 2
                dx_eta = (eta_r - eta_l) / 2
                theta_r = (-dx_eta + np.sqrt(dx_eta**2 - 4 * dxx_eta * eta)) / (
                    2 * dxx_eta
                )
            else:
                ...
        if eta * eta_b < 0:
            ...
        if eta * eta_t < 0:
            ...

        x_l, y_l = x - theta_l * dx, y
        x_r, y_r = x + theta_r * dx, y
        x_b, y_b = x, y - theta_b * dy
        x_t, y_t = x, y + theta_t * dy

        a_tau = -n2 * (
            -a[i + 2, j] + 8 * a[i + 1, j] - 8 * a[i - 1, j] + a[i - 2, j]
        ) / (12 * dx) + n1 * (
            -a[i, j + 2] + 8 * a[i, j + 1] - 8 * a[i, j - 1] + a[i, j - 2]
        ) / (
            12 * dy
        )

        # TODO: figure out N and D

        # solve for u-

        # matrix coeff and rho can be obtained

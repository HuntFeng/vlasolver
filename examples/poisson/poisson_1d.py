# 1D second-order interface Poisson solver demonstration
# Implements a simplified version of Section 3.1 (one interface, piecewise-constant beta)
# and compares with exact solution, then plots.

import matplotlib.pyplot as plt
import numpy as np

M_EPS = 1e-6


class PoissonProblem:
    def __init__(self, nx: int, xI: float) -> None:
        self.xI = xI
        self.nx = 10
        self.dx = 1 / nx
        self.x = np.arange(self.dx / 2, 1 + self.dx / 2, self.dx)
        self.a = np.zeros_like(self.x)  # [u]
        self.b = np.zeros_like(self.x)  # [beta u_x]
        self.u_exact = np.sin(np.pi * self.x)
        self.f = -np.pi**2 * np.sin(np.pi * self.x)

    def surface(self, x: float):
        return x - self.xI

    def permittivity(self, x: float):
        return 1.0

    def compute_theta(self, i: int) -> float:
        x = self.x
        eta = self.surface(x[i])
        eta_r = self.surface(x[i + 1])
        eta_l = self.surface(x[i - 1])
        dx_eta = (eta_r - eta_l) / 2
        dxx_eta = (eta_r - 2 * eta + eta_l) / 2

        if np.isclose(dxx_eta, 0.0):
            theta = np.abs(eta / dx_eta)
        else:
            theta = (
                -dx_eta - np.sign(eta) * np.sqrt(dx_eta**2 - 4 * dxx_eta * eta)
            ) / (2.0 * dxx_eta)
        return theta

    def construct_matrix(self):
        nx = self.nx
        u_exact = self.u_exact
        x = self.x
        f = self.f
        dx = self.dx
        a = self.a
        b = self.b
        xI = self.xI

        A = np.zeros((nx, nx))
        A[0, 0] = 1.0
        A[-1, -1] = 1.0

        f[0] = u_exact[0]
        f[-1] = u_exact[-1]

        for i in range(1, nx - 1):
            eta = self.surface(x[i])
            eta_l = self.surface(x[i - 1])
            eta_r = self.surface(x[i + 1])
            theta = self.compute_theta(i)

            if eta > 0.0:
                eps_p = self.permittivity(xI - M_EPS)
                eps_m = self.permittivity(xI + M_EPS)
            else:
                eps_p = self.permittivity(xI + M_EPS)
                eps_m = self.permittivity(xI - M_EPS)

            eps_1 = eps_p if eta > 0.0 else eps_m
            eps_2 = eps_m if eta > 0.0 else eps_p
            eps_hat = eps_2 * (3 - 2 * theta) / ((1 - theta) * (2 - theta)) + eps_1 * (
                2 * theta + 1
            ) / (theta * (theta + 1))
            C = np.array(
                [
                    eps_1 * theta / (theta + 1),
                    -eps_1 * (theta + 1) / theta,
                    -eps_2 * (2 - theta) / (1 - theta),
                    eps_2 * (1 - theta) / (2 - theta),
                ]
            )

            bot = (theta + 1) / 2 * dx**2
            if eta * eta_r < 0.0:
                eps_l = self.permittivity(x[i] - dx / 2)
                eps_r = self.permittivity(x[i] + theta * dx / 2 - M_EPS)
                A[i, i - 1] = -C[0] / eps_hat / theta * eps_r / bot + eps_l / bot
                A[i, i] = (
                    -C[1] / eps_hat / theta * eps_r / bot
                    - eps_r / theta / bot
                    - eps_l / bot
                )
                A[i, i + 1] = -C[2] / eps_hat / theta * eps_r / bot
                A[i, i + 2] = -C[3] / eps_hat / theta * eps_r / bot
                f[i] -= (
                    np.sign(eta)
                    / eps_hat
                    * (
                        eps_p * a[i] * (3 - 2 * theta) / ((1 - theta) * (2 - theta))
                        + b[i] * dx
                    )
                )
            elif eta * eta_l < 0.0:
                eps_l = self.permittivity(x[i] - theta * dx / 2 + M_EPS)
                eps_r = self.permittivity(x[i] + dx / 2)
                A[i, i + 1] = -C[0] / eps_hat / theta * eps_l / bot + eps_r / bot
                A[i, i] = (
                    -C[1] / eps_hat / theta * eps_l / bot
                    - eps_r / bot
                    - eps_l / theta / bot
                )
                A[i, i - 1] = -C[2] / eps_hat / theta * eps_l / bot
                A[i, i - 2] = -C[3] / eps_hat / theta * eps_l / bot

                f[i] -= (
                    np.sign(eta)
                    / eps_hat
                    * (
                        eps_p * a[i] * (3 - 2 * theta) / ((1 - theta) * (2 - theta))
                        - b[i] * dx
                    )
                )
            else:
                eps_l = self.permittivity(x[i] - dx / 2)
                eps_r = self.permittivity(x[i] + dx / 2)
                A[i, i] = -(eps_r + eps_l) / dx**2
                A[i, i + 1] = eps_r / dx**2
                A[i, i - 1] = eps_l / dx**2
        return A

    def solve(self):
        self.A = self.construct_matrix()
        self.u = np.linalg.solve(self.A, self.f)


p1 = PoissonProblem(nx=10, xI=0.5)
p1.solve()
plt.figure()
plt.spy(p1.A)
plt.title(f"$x_I = {p1.xI}$")


p2 = PoissonProblem(nx=10, xI=0.8)
p2.solve()
plt.figure()
plt.spy(p2.A)
plt.title(f"$x_I = {p2.xI}$")

# plot
plt.figure()
plt.plot(p1.x, p1.u_exact, label="Exact", linewidth=2)
plt.plot(p1.x, p1.u, "--", label="Numerical 1")
plt.plot(p1.x, p2.u, "--", label="Numerical 2")
# plt.axvline(xI)
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("1D Second-Order Interface Poisson Solver")
plt.show()

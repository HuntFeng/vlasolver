from typing import Callable

import numpy as np


class GaussSeidel:
    def __init__(self, origin=[0.0, 0.0], L=[1.0, 0.5], n_int=[100, 50], tol=1e-6):
        self.nghost = 3  # Number of ghost cells
        G = self.nghost
        self.max_iter = 5000
        self.L = L
        self.origin = origin
        n = [n_int[0] + 2 * G, n_int[1] + 2 * G]
        self.eps = np.ones(n)
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.u = np.zeros(n)
        self.surface: Callable[..., float]
        self.normal: Callable[..., tuple[float, float]]
        self.tol = tol

    def apply_boundary(self, u):
        """Apply boundary conditions to the solution array."""
        # G = self.nghost
        # dx = self.L[0] / (u.shape[0] - 2 * G)
        # dy = self.L[1] / (u.shape[1] - 2 * G)
        # for i in np.arange(u.shape[0]):
        #     for j in np.arange(u.shape[1]):
        #         if i >= G and i < u.shape[0] - G and j >= G and j < u.shape[1] - G:
        #             continue
        #         x = self.origin[0] + dx * (i - G + 0.5)
        #         y = self.origin[1] + dy * (j - G + 0.5)
        #         u[i, j] = np.exp(x**2 + y**2) / eps_p
        return u

    def compute_error(self, u_old, u):
        """Compute the error between the current solution and the source term."""
        G = self.nghost
        return np.abs(u - u_old)[G:-G, G:-G].max()

    def red_black_update(self, u, g, is_update_red: int):
        G = self.nghost
        omega = 1.5
        dx = self.L[0] / (u.shape[0] - 2 * G)
        dy = self.L[1] / (u.shape[1] - 2 * G)
        eps = self.eps
        a = self.a
        b = self.b

        for i in np.arange(G, u.shape[0] - G):
            for j in np.arange(G, u.shape[1] - G):
                if (i + j) % 2 == is_update_red:
                    # compute x, y coordinates
                    x = self.origin[0] + dx * (i - G + 0.5)
                    y = self.origin[1] + dy * (j - G + 0.5)

                    # jump condition term at left, right, top, bottom fluxes
                    F_l, F_r, F_b, F_t = 0.0, 0.0, 0.0, 0.0

                    # eps at left, right, top, bottom fluxes
                    eps_l = 0.5 * (eps[i - 1, j] + eps[i, j])
                    eps_r = 0.5 * (eps[i + 1, j] + eps[i, j])
                    eps_b = 0.5 * (eps[i, j - 1] + eps[i, j])
                    eps_t = 0.5 * (eps[i, j + 1] + eps[i, j])

                    # interior indicator
                    eta = self.surface(x, y)
                    eta_l = self.surface(x - dx, y)
                    eta_r = self.surface(x + dx, y)
                    eta_b = self.surface(x, y - dy)
                    eta_t = self.surface(x, y + dy)

                    # modify eps and F if discontinuity is detected
                    if eta * eta_l <= 0.0:
                        eta_p = eta if (eta > 0.0) else eta_l
                        eta_m = eta if (eta <= 0.0) else eta_l
                        eps_p = eps[i, j] if (eta > 0.0) else eps[i - 1, j]
                        eps_m = eps[i, j] if (eta <= 0.0) else eps[i - 1, j]
                        eps_l = (
                            eps_p
                            * eps_m
                            * (abs(eta_m) + abs(eta_p))
                            / (eps_p * abs(eta_m) + eps_m * abs(eta_p))
                        )

                        n1, n2 = self.normal(x, y, dx, dy)
                        n1_l, n2_l = self.normal(x - dx, y, dx, dy)
                        theta = abs(eta_l) / (abs(eta) + abs(eta_l))
                        a_gamma = (a[i, j] * abs(eta_l) + a[i - 1, j] * abs(eta)) / (
                            abs(eta) + abs(eta_l)
                        )
                        b_gamma = (
                            b[i, j] * n1 * abs(eta_l) + b[i - 1, j] * n1_l * abs(eta)
                        ) / (abs(eta) + abs(eta_l))
                        if eta <= 0.0:
                            F_l = eps_l * a_gamma / (
                                dx * dx
                            ) - eps_l * b_gamma * theta / (eps_p * dx)
                        else:
                            F_l = -eps_l * a_gamma / (
                                dx * dx
                            ) + eps_l * b_gamma * theta / (eps_m * dx)
                    if eta * eta_r <= 0.0:
                        eta_p = eta if (eta > 0.0) else eta_r
                        eta_m = eta if (eta <= 0.0) else eta_r
                        eps_p = eps[i, j] if (eta > 0.0) else eps[i + 1, j]
                        eps_m = eps[i, j] if (eta <= 0.0) else eps[i + 1, j]
                        eps_r = (
                            eps_p
                            * eps_m
                            * (abs(eta_m) + abs(eta_p))
                            / (eps_p * abs(eta_m) + eps_m * abs(eta_p))
                        )

                        n1, n2 = self.normal(x, y, dx, dy)
                        n1_r, n2_r = self.normal(x + dx, y, dx, dy)
                        theta = abs(eta_r) / (abs(eta) + abs(eta_r))
                        a_gamma = (a[i, j] * abs(eta_r) + a[i + 1, j] * abs(eta)) / (
                            abs(eta) + abs(eta_r)
                        )
                        b_gamma = (
                            b[i, j] * n1 * abs(eta_r) + b[i + 1, j] * n1_r * abs(eta)
                        ) / (abs(eta) + abs(eta_r))
                        if eta <= 0.0:
                            F_r = eps_r * a_gamma / (
                                dx * dx
                            ) + eps_r * b_gamma * theta / (eps_p * dx)
                        else:
                            F_r = -eps_r * a_gamma / (
                                dx * dx
                            ) - eps_r * b_gamma * theta / (eps_m * dx)
                    if eta * eta_b <= 0.0:
                        eta_p = eta if (eta > 0.0) else eta_b
                        eta_m = eta if (eta <= 0.0) else eta_b
                        eps_p = eps[i, j] if (eta > 0.0) else eps[i, j - 1]
                        eps_m = eps[i, j] if (eta <= 0.0) else eps[i, j - 1]
                        eps_b = (
                            eps_p
                            * eps_m
                            * (abs(eta_m) + abs(eta_p))
                            / (eps_p * abs(eta_m) + eps_m * abs(eta_p))
                        )

                        n1, n2 = self.normal(x, y, dx, dy)
                        n1_b, n2_b = self.normal(x, y - dy, dx, dy)
                        theta = abs(eta_b) / (abs(eta) + abs(eta_b))
                        a_gamma = (a[i, j] * abs(eta_b) + a[i, j - 1] * abs(eta)) / (
                            abs(eta) + abs(eta_b)
                        )
                        b_gamma = (
                            b[i, j] * n2 * abs(eta_b) + b[i, j - 1] * n2_b * abs(eta)
                        ) / (abs(eta) + abs(eta_b))
                        if eta <= 0.0:
                            F_b = eps_b * a_gamma / (
                                dy * dy
                            ) - eps_b * b_gamma * theta / (eps_p * dy)
                        else:
                            F_b = -eps_b * a_gamma / (
                                dy * dy
                            ) + eps_b * b_gamma * theta / (eps_m * dy)
                    if eta * eta_t <= 0.0:
                        eta_p = eta if (eta > 0.0) else eta_t
                        eta_m = eta if (eta <= 0.0) else eta_t
                        eps_p = eps[i, j] if (eta > 0.0) else eps[i, j + 1]
                        eps_m = eps[i, j] if (eta <= 0.0) else eps[i, j + 1]
                        eps_t = (
                            eps_p
                            * eps_m
                            * (abs(eta_m) + abs(eta_p))
                            / (eps_p * abs(eta_m) + eps_m * abs(eta_p))
                        )

                        n1, n2 = self.normal(x, y, dx, dy)
                        n1_t, n2_t = self.normal(x, y + dy, dx, dy)
                        theta = abs(eta_t) / (abs(eta) + abs(eta_t))
                        a_gamma = (a[i, j] * abs(eta_t) + a[i, j + 1] * abs(eta)) / (
                            abs(eta) + abs(eta_t)
                        )
                        b_gamma = (
                            b[i, j] * n2 * abs(eta_t) + b[i, j + 1] * n2_t * abs(eta)
                        ) / (abs(eta) + abs(eta_t))
                        if eta <= 0.0:
                            F_t = eps_t * a_gamma / (
                                dy * dy
                            ) + eps_t * b_gamma * theta / (eps_p * dy)
                        else:
                            F_t = -eps_t * a_gamma / (
                                dy * dy
                            ) - eps_t * b_gamma * theta / (eps_m * dy)

                    # update potential field
                    denom = (eps_l + eps_r) / (dx * dx) + (eps_b + eps_t) / (dy * dy)
                    average = (eps_l * u[i - 1, j] + eps_r * u[i + 1, j]) / (
                        dx * dx
                    ) + (eps_b * u[i, j - 1] + eps_t * u[i, j + 1]) / (dy * dy)
                    Fx = F_l + F_r
                    Fy = F_b + F_t

                    # under-relaxation update
                    u[i, j] = (1 - omega) * u[i, j] + omega * (
                        average - g[i, j] - Fx - Fy
                    ) / denom

    def solve(self, u, g):
        """Perform Gauss-Seidel iteration."""
        for iter in range(self.max_iter):
            u_old = u.copy()
            self.red_black_update(u, g, is_update_red=0)
            self.red_black_update(u, g, is_update_red=1)

            u = self.apply_boundary(u)
            error = self.compute_error(u_old, u)
            print(f"Iteration {iter + 1}, max error: {error}")
            if error < self.tol:
                print(f"Converged after {iter + 1} iterations.")
                break
        # return u[G:-G, G:-G]
        return u


def example1():
    nx, ny = 128, 128
    # nx, ny = 64, 64
    # nx, ny = 8, 8
    G = 3  # Number of ghost cells

    eps_m = 1.0
    eps_p = 10.0
    origin = [-1, -1]
    L = [2.0, 2.0]
    n_int = [nx, ny]

    dx = 2.0 / nx
    dy = 2.0 / ny
    x = np.arange(origin[0] + dx / 2, 1.0, dx)
    y = np.arange(origin[1] + dy / 2, 1.0, dy)
    Y, X = np.meshgrid(y, x)

    gs = GaussSeidel(origin=origin, L=L, n_int=n_int)
    r0 = np.pi / 6.28
    gs.surface = lambda x, y: x**2 + y**2 - r0**2
    gs.normal = lambda x, y, dx, dy: (
        x / np.sqrt(x**2 + y**2),
        y / np.sqrt(x**2 + y**2),
    )

    def apply_boundary(u):
        """Apply Dirichlet"""
        G = gs.nghost
        dx = gs.L[0] / (u.shape[0] - 2 * G)
        dy = gs.L[1] / (u.shape[1] - 2 * G)
        for i in np.arange(u.shape[0]):
            for j in np.arange(u.shape[1]):
                if i >= G and i < u.shape[0] - G and j >= G and j < u.shape[1] - G:
                    continue
                x = gs.origin[0] + dx * (i - G + 0.5)
                y = gs.origin[1] + dy * (j - G + 0.5)
                u[i, j] = np.exp(x**2 + y**2) / eps_p
        return u

    gs.apply_boundary = apply_boundary

    gs.eps[:, :] = eps_p
    gs.eps[G:-G, G:-G][X**2 + Y**2 <= r0**2] = eps_m
    # gs.eps[G:-G, G:-G][X**2 + Y**2 > r0**2] = eps_p
    gs.a[:, :] = 0.0
    gs.b[:, :] = 0.0
    g = np.zeros_like(gs.u)
    g[G:-G, G:-G] = 4 * (1 + X**2 + Y**2) * np.exp(X**2 + Y**2)
    u = gs.apply_boundary(gs.u)
    u = gs.solve(u, g)

    plt.figure()
    plt.pcolormesh(X, Y, u, cmap="jet")
    plt.colorbar(label="$\\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential Field")
    plt.show()


def example2():
    """Example 3 from XuDong Liu 2000"""
    nx, ny = 64, 64
    # nx, ny = 32, 32
    origin = [0.0, 0.0]
    L = [1.0, 1.0]
    dx = L[0] / nx
    dy = L[1] / ny
    x = np.arange(origin[0] - 2.5 * dx, origin[0] + L[0] + 3 * dx, dx)
    y = np.arange(origin[1] - 2.5 * dy, origin[1] + L[1] + 3 * dx, dy)
    Y, X = np.meshgrid(y, x)

    r0 = 0.25
    gs = GaussSeidel(origin=origin, L=L, n_int=[nx, ny])
    gs.surface = lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 - r0**2
    gs.normal = lambda x, y, dx, dy: (
        (2 * x - 1) / np.sqrt((2 * x - 1) ** 2 + (2 * y - 1) ** 2),
        (2 * y - 1) / np.sqrt((2 * x - 1) ** 2 + (2 * y - 1) ** 2),
    )

    def apply_boundary(u):
        """Apply Dirichlet"""
        G = gs.nghost
        u[:G, :] = 0.0
        u[-G:, :] = 0.0
        u[:, :G] = 0.0
        u[:, -G:] = 0.0
        return u

    gs.apply_boundary = apply_boundary
    gs.eps[:, :] = 1.0
    gs.eps[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2] = 2.0
    gs.a[:, :] = -np.exp(-(X**2) - Y**2)
    gs.b[:, :] = 8 * (2 * X**2 + 2 * Y**2 - X - Y) * np.exp(-(X**2) - Y**2)
    g = np.zeros_like(gs.u)
    g[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2] = (
        8 * (X**2 + Y**2 - 1) * np.exp(-(X**2) - Y**2)
    )[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2]
    u = gs.apply_boundary(gs.u)
    # u = gs.solve(u, g)
    u_exact = np.zeros_like(X)
    u_exact[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2] = (np.exp(-(X**2) - Y**2))[
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2
    ]

    G = gs.nghost
    for iter in range(1000):
        gs.red_black_update(u, g, is_update_red=0)
        gs.red_black_update(u, g, is_update_red=1)

        u = gs.apply_boundary(u)
        error = np.max(np.abs(u - u_exact)[G:-G, G:-G])
        print(f"Iteration {iter + 1}, max error: {error}")
        if error < 1e-2:
            print(f"Converged after {iter + 1} iterations.")
            break
    print(f"Max error: {np.max(np.abs(u-u_exact)[G:-G, G:-G]):.2e}")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = ax.scatter(X, Y, u, c=u, cmap="coolwarm")
    # scatter = ax.scatter(X, Y, u_exact, c=u_exact, cmap="coolwarm")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(scatter, ax=ax, label="$\\phi$")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # example1()
    example2()

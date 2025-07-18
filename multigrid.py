import numpy as np


class Multigrid:
    """
    2D Cell-centered Multigrid solver for the nonlinear Poisson equation using Full Approximation Scheme (FAS):
        u_xx + u_yy + f(u) = g in (a, b)x(c, d),
        Dirichlet on left, Neumann (zero-gradient) on top/bottom/right.
        The grid includes 3 ghost cells on each side (cell-centered).
    """

    def __init__(
        self,
        f,
        g,
        origin=[0.0, 0.0],
        L=[10.0, 10.0],
        n_phys=[64, 64],
        levels=4,
        tol=1e-6,
        max_iter=100,
    ):
        """
        Args:
            f: function, the nonlinear term f(u), should be vectorized.
            g: array, the right-hand side g(x), shape: n_phys
            L: list, physical domain size
            n_phys: int list, number of cells [Nx, Ny] (excluding ghosts)
            levels: int, number of multigrid levels
            tol: float, stopping tolerance
            max_iter: int, maximum number of V-cycles
        """
        self.f = f
        self.g = g
        self.L = L
        self.origin = origin
        self.n_phys = n_phys
        self.levels = levels
        self.tol = tol
        self.max_iter = max_iter
        # Total size including 3 ghost cells per side
        self.nghost = 3
        self.n = [n_phys[0] + 2 * self.nghost, n_phys[1] + 2 * self.nghost]
        self.u = np.zeros(self.n, dtype=float)
        self.err = 100.0

    def surface(self, x, y) -> float:
        return 0.0

    def normal(self, x, y, dx, dy) -> tuple[float, float]:
        return (0.0, 0.0)

    def apply_boundary(self, u) -> np.ndarray:
        return u

    def construct_jump_conditions(self, u) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return u, u, u

    def gauss_seidel(self, u, g, eps, a, b, iter=3):
        """Nonlinear Gauss-Seidel smoothing with under-relaxation."""

        dx = self.L[0] / (u.shape[0] - 2 * self.nghost)
        dy = self.L[1] / (u.shape[1] - 2 * self.nghost)
        x = np.arange(
            self.origin[0] - 2.5 * dx, self.origin[0] + self.L[0] + 3 * dx, dx
        )
        y = np.arange(
            self.origin[1] - 2.5 * dy, self.origin[1] + self.L[1] + 3 * dx, dy
        )
        Y, X = np.meshgrid(y, x)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        scatter = ax.scatter(X, Y, u, c=u, cmap="coolwarm")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Potential before smoothing")
        plt.colorbar(scatter, ax=ax, label="$\\phi$")
        plt.tight_layout()
        plt.show()
        G = self.nghost
        for n in range(iter):
            u_old = u.copy()
            self.red_black_update(u, g, eps, a, b, is_update_red=0)
            self.red_black_update(u, g, eps, a, b, is_update_red=1)
            u = self.apply_boundary(u)
            self.err = np.max(np.abs(u - u_old)[G : -G, G : -G])

            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # scatter = ax.scatter(X, Y, u, c=u, cmap="coolwarm")
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_title(f"Potential after {n}th Smoothing")
            # plt.colorbar(scatter, ax=ax, label="$\\phi$")
            # plt.tight_layout()
            # plt.show()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        scatter = ax.scatter(X, Y, u, c=u, cmap="coolwarm")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Potential after {iter}th Smoothing")
        plt.colorbar(scatter, ax=ax, label="$\\phi$")
        plt.tight_layout()
        plt.show()
        return u

    def red_black_update(self, u, g, eps, a, b, is_update_red: int):
        omega = 1.5
        # omega = 0.5
        G = self.nghost
        dx = self.L[0] / (u.shape[0] - 2 * G)
        dy = self.L[1] / (u.shape[1] - 2 * G)
        for i in np.arange(G, u.shape[0] - G):
            for j in np.arange(G, u.shape[1] - G):
                if (i-G + j-G) % 2 == is_update_red:
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
                            # F_r = -eps_r * a_gamma / (
                            #     dx * dx
                            # ) + eps_r * b_gamma * theta / (eps_p * dx)
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
                            # F_b = +eps_b * a_gamma / (
                            #     dy * dy
                            # ) + eps_b * b_gamma * theta / (eps_p * dy)
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
                        average - g[i, j] + self.f(u[i, j]) - Fx - Fy
                    ) / denom

    def nonlinear_operator(self, u, eps):
        """N(u) = u_xx + u_yy + f(u), returns N(u)."""
        G = self.nghost
        dx = self.L[0] / (u.shape[0] - 2 * G)
        dy = self.L[1] / (u.shape[1] - 2 * G)
        out = np.zeros_like(u)
        f = self.f

        for i in np.arange(G, u.shape[0] - G):
            for j in np.arange(G, u.shape[1] - G):
                # compute x, y coordinates
                x = self.origin[0] + dx * (i - G + 0.5)
                y = self.origin[1] + dy * (j - G + 0.5)

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

                ddx = (
                    eps_r * (u[i + 1, j] - u[i, j]) - eps_l * (u[i, j] - u[i - 1, j])
                ) / dx**2
                ddy = (
                    eps_t * (u[i, j + 1] - u[i, j]) - eps_b * (u[i, j] - u[i, j - 1])
                ) / dy**2

                out[i, j] = ddx + ddy #+ f(u[i, j])
        return out

    def restrict(self, u_fine):
        """Restrict to coarse grid (cell-centered)."""
        G = self.nghost
        nf_x, nf_y = u_fine.shape
        nc_x = (nf_x - 2 * G) // 2 + 2 * G
        nc_y = (nf_y - 2 * G) // 2 + 2 * G
        u_coarse = np.zeros((nc_x, nc_y))

        for ic in np.arange(G, nc_x - G):
            for jc in np.arange(G, nc_y - G):
                i_f = 2 * (ic - G) + G
                j_f = 2 * (jc - G) + G
                u_coarse[ic, jc] = 0.25 * (
                    u_fine[i_f, j_f]
                    + u_fine[i_f + 1, j_f]
                    + u_fine[i_f, j_f + 1]
                    + u_fine[i_f + 1, j_f + 1]
                )
                # x_c = self.origin[0] + (ic - G + 0.5) * (self.L[0] / (nc_x - 2 * G))
                # y_c = self.origin[1] + (jc - G + 0.5) * (self.L[1] / (nc_y - 2 * G))
                #
                # is_coarse_internal = self.surface(x_c, y_c) > 0.0
                #
                # fine_indices = [
                #     (2 * (ic - G) + G, 2 * (jc - G) + G),
                #     (2 * (ic - G) + G + 1, 2 * (jc - G) + G),
                #     (2 * (ic - G) + G, 2 * (jc - G) + G + 1),
                #     (2 * (ic - G) + G + 1, 2 * (jc - G) + G + 1),
                # ]
                # values = []
                # for i_f, j_f in fine_indices:
                #     x_f = self.origin[0] + (i_f - G + 0.5) * (
                #         self.L[0] / (nf_x - 2 * G)
                #     )
                #     y_f = self.origin[1] + (j_f - G + 0.5) * (
                #         self.L[1] / (nf_y - 2 * G)
                #     )
                #     is_fine_internal = self.surface(x_f, y_f) > 0.0
                #     if is_coarse_internal == is_fine_internal:
                #         values.append(u_fine[i_f, j_f])
                # if len(values) == 0:
                #     u_coarse[ic, jc] = 0.0
                # else:
                #     u_coarse[ic, jc] = np.mean(values)

        # plt.figure()
        # plt.imshow(u_coarse, cmap="jet")
        # plt.xlabel("i")
        # plt.ylabel("j")
        # plt.title("Restricted Coarse Grid Potential Field")
        # plt.show()
        return u_coarse

    def prolong(self, ec, n_fine):
        """Prolongate error to finer grid using bilinear interpolation (cell-centered)."""
        ef = np.zeros(n_fine)
        G = self.nghost
        nf_x, nf_y = n_fine

        for i in np.arange(G, nf_x - G):
            for j in np.arange(G, nf_y - G):
                ic = (i - G) // 2 + G
                jc = (j - G) // 2 + G
                ef[i, j] = ec[ic, jc]
                # if (i - G) % 2 == 0 and (j - G) % 2 == 0:
                #     ef[i, j] = ec[ic, jc]
                # elif (i - G) % 2 == 0 and (j - G) % 2 == 1:
                #     ef[i, j] = 0.5 * (ec[ic, jc] + ec[ic, jc + 1])
                # elif (i - G) % 2 == 1 and (j - G) % 2 == 0:
                #     ef[i, j] = 0.5 * (ec[ic, jc] + ec[ic + 1, jc])
                # elif (i - G) % 2 == 1 and (j - G) % 2 == 1:
                #     ef[i, j] = 0.25 * (
                #         ec[ic, jc]
                #         + ec[ic + 1, jc]
                #         + ec[ic, jc + 1]
                #         + ec[ic + 1, jc + 1]
                #     )

        # print("Prolongated fine grid ef[2,2]:", ef[2, 2])
        # plt.figure()
        # plt.imshow(ef, cmap="jet")
        # plt.colorbar(label="$e_{fine}$")
        # plt.xlabel("i")
        # plt.ylabel("j")
        # plt.title("Prolongated Fine Grid Error Field")
        # plt.show()
        return ef

    def fas_v_cycle(self, level, u, g, eps, a, b):
        """Full Approximation Scheme (FAS) V-cycle."""
        print(f"------------------------V-cycle level {level}, shape: {u.shape}")
        print("pre-smoothing")
        u = self.gauss_seidel(u, g, eps, a, b, iter=20)
        if level == self.levels - 1:
            print("bottom level reached, applying final smoothing")
            u = self.gauss_seidel(u, g, eps, a, b, iter=30)
            return u
        print("calculating nonlinear operator")
        lhs = self.nonlinear_operator(u, eps)
        print("restricting u to coarse grid")
        u_c = self.restrict(u)
        print("restricting lhs to coarse grid")
        lhs_c = self.restrict(lhs)
        print("restricting g to coarse grid")
        g_c = self.restrict(g)
        print("applying boundary condition to u_c")
        u_c = self.apply_boundary(u_c)
        eps_c, a_c, b_c = self.construct_jump_conditions(u_c)

        # plt.figure()
        # plt.imshow(u_c, cmap="jet")
        # plt.colorbar(label="$u_{coarse}$")
        # plt.xlabel("i")
        # plt.ylabel("j")
        # plt.title("Restrict to Coarse Grid Potential Field")
        # plt.show()
        u_c_old = u_c.copy()
        print("calculating nonlinear operator")
        tau_c = lhs_c - self.nonlinear_operator(u_c, eps_c)
        g_fas = g_c + tau_c
        # plt.figure()
        # plt.imshow(g_fas, cmap="jet")
        # plt.colorbar(label="$g_{fas}$")
        # plt.xlabel("i")
        # plt.ylabel("j")
        # plt.title("FAS Right-Hand Side Field")
        # plt.show()
        u_c = self.fas_v_cycle(level + 1, u_c, g_fas, eps_c, a_c, b_c)
        print("prolonging correction to fine grid")
        corr = self.prolong(u_c - u_c_old, u.shape)
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(u_c_old, cmap="jet")
        # ax[0].set_title("Coarse Grid Potential Field (Old)")
        # ax[0].set_xlabel("i")
        # ax[0].set_ylabel("j")
        # ax[1].imshow(u_c, cmap="jet")
        # ax[1].set_title("Coarse Grid Potential Field")
        # ax[1].set_xlabel("i")
        # ax[1].set_ylabel("j")
        # ax[2].imshow(corr, cmap="jet")
        # ax[2].set_title("Fine Grid Correction Field")
        # ax[2].set_xlabel("i")
        # ax[2].set_ylabel("j")
        # plt.tight_layout()
        # plt.show()
        u += corr
        print("applying boundary condition to fine u")
        u = self.apply_boundary(u)
        eps, a, b = self.construct_jump_conditions(u)
        print("post-smoothing")
        u = self.gauss_seidel(u, g, eps, a, b, iter=10)
        return u

    def solve(self, u=None):
        """Solve the nonlinear Poisson's equation with FAS multigrid (cell-centered, 3 ghost cells, Dirichlet/Neumann BCs)."""
        if u is None:
            u = np.zeros(self.n)
        g = self.g
        u = self.apply_boundary(u)
        eps, a, b = self.construct_jump_conditions(u)
        G = self.nghost
        for i in range(self.max_iter):
            u_old = u.copy()
            u = self.fas_v_cycle(0, u, g, eps, a, b)
            err = np.max(np.abs(u - u_old)[G:-G, G:-G])
            # print(f"Iteration {i + 1}, max error: {err:.2e}")
            print(f"Iteration {i + 1}, max error: {self.err:.2e}")
            if self.err < self.tol:
                print(f"Converged after {i + 1} V-cycles with error {self.err:.2e}")
                break
            # if err < self.tol:
            #     print(f"Converged after {i + 1} V-cycles with error {err:.2e}")
            #     break
        return u


def example1():
    """Example 1 from Hongtao Liu 2023"""
    eps_m = 1.0
    eps_p = 10.0

    nx, ny = 128, 128
    origin = [-1.0, -1.0]
    L = [2.0, 2.0]
    dx = L[0] / nx
    dy = L[1] / ny
    x = np.arange(origin[0] - 2.5 * dx, origin[0] + L[0] + 3 * dx, dx)
    y = np.arange(origin[1] - 2.5 * dy, origin[1] + L[1] + 3 * dx, dy)
    Y, X = np.meshgrid(y, x)
    g = 4 * (1 + X**2 + Y**2) * np.exp(X**2 + Y**2)
    f = lambda u: 0

    mg = Multigrid(f, g, origin=origin, L=L, n_phys=[nx, ny], levels=4, tol=1e-3)
    r0 = np.pi / 6.28
    mg.surface = lambda x, y: x**2 + y**2 - r0**2
    mg.normal = lambda x, y, dx, dy: (
        x / np.sqrt(x**2 + y**2),
        y / np.sqrt(x**2 + y**2),
    )

    def apply_boundary(u):
        """Apply Dirichlet (left/right) and Neumann (top/bottom) BCs on 3 ghost cells."""
        G = mg.nghost
        dx = mg.L[0] / (u.shape[0] - 2 * G)
        dy = mg.L[1] / (u.shape[1] - 2 * G)
        for i in np.arange(u.shape[0]):
            for j in np.arange(u.shape[1]):
                if i >= G and i < u.shape[0] - G and j >= G and j < u.shape[1] - G:
                    continue
                x = mg.origin[0] + dx * (i - G + 0.5)
                y = mg.origin[1] + dy * (j - G + 0.5)
                u[i, j] = np.exp(x**2 + y**2) / eps_p
        return u

    def construct_jump_conditions(u):
        """Reconstruct permittivity and jump conditions on coarse grid."""
        eps = np.zeros_like(u)
        a = np.zeros_like(u)
        b = np.zeros_like(u)
        G = mg.nghost
        dx = mg.L[0] / (u.shape[0] - 2 * G)
        dy = mg.L[1] / (u.shape[1] - 2 * G)
        for i in np.arange(0, u.shape[0]):
            for j in np.arange(0, u.shape[1]):
                x = mg.origin[0] + (i - G + 0.5) * dx
                y = mg.origin[1] + (j - G + 0.5) * dy
                eta = mg.surface(x, y)
                if eta <= 0.0:
                    eps[i, j] = eps_m
                else:
                    eps[i, j] = eps_p
        return eps, a, b

    mg.apply_boundary = apply_boundary
    mg.construct_jump_conditions = construct_jump_conditions

    u = mg.solve()
    u_exact = np.exp(X**2 + Y**2) / eps_p
    u_exact[X**2 + Y**2 <= r0**2] = (
        np.exp(X**2 + Y**2) / eps_m + (1 / eps_p - 1 / eps_m) * np.exp(r0**2)
    )[X**2 + Y**2 <= r0**2]
    G = mg.nghost
    np.max(np.abs(u - u_exact)[G:-G, G:-G])
    print(f"Error: {mg.err:.2e}")

    plt.figure()
    plt.pcolormesh(X[G:-G, G:-G], Y[G:-G, G:-G], u[G:-G, G:-G], cmap="jet")
    plt.colorbar(label="$\\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential Field")
    plt.show()


# FIXME: This example is not working properly, we can't get nice discontinuity if levels > 1...
# It seems the grid coasening losses the discontinuity information even if I use conceptually working prolongation and restriction algo.
def example3():
    """Example 3 from XuDong Liu 2000"""
    nx, ny = 64, 64
    origin = [0.0, 0.0]
    L = [1.0, 1.0]
    dx = L[0] / nx
    dy = L[1] / ny
    x = np.arange(origin[0] - 2.5 * dx, origin[0] + L[0] + 3 * dx, dx)
    y = np.arange(origin[1] - 2.5 * dy, origin[1] + L[1] + 3 * dx, dy)
    Y, X = np.meshgrid(y, x)

    r0 = 0.25
    g = np.zeros_like(X)
    g[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2] = (
        8 * (X**2 + Y**2 - 1) * np.exp(-(X**2) - Y**2)
    )[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2]
    f = lambda u: 0

    # FIXME: error does not go down after first few interation if levels > 1
    mg = Multigrid(f, g, origin=origin, L=L, n_phys=[nx, ny], levels=4, tol=1e-3)
    mg.max_iter = 20
    mg.surface = lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 - r0**2
    mg.normal = lambda x, y, dx, dy: (
        (2 * x - 1) / np.sqrt((2 * x - 1) ** 2 + (2 * y - 1) ** 2),
        (2 * y - 1) / np.sqrt((2 * x - 1) ** 2 + (2 * y - 1) ** 2),
    )

    def apply_boundary(u):
        """Apply Dirichlet"""
        G = mg.nghost
        u[:G, :] = 0.0
        u[-G:, :] = 0.0
        u[:, :G] = 0.0
        u[:, -G:] = 0.0
        return u

    def construct_jump_conditions(u):
        """Reconstruct permittivity and jump conditions on coarse grid."""
        eps = np.zeros_like(u)
        a = np.zeros_like(u)
        b = np.zeros_like(u)
        G = mg.nghost
        dx = mg.L[0] / (u.shape[0] - 2 * G)
        dy = mg.L[1] / (u.shape[1] - 2 * G)
        for i in np.arange(0, u.shape[0]):
            for j in np.arange(0, u.shape[1]):
                x = mg.origin[0] + (i - G + 0.5) * dx
                y = mg.origin[1] + (j - G + 0.5) * dy
                eta = mg.surface(x, y)
                a[i, j] = -np.exp(-(x**2) - y**2)
                b[i, j] = 8 * (2 * x**2 + 2 * y**2 - x - y) * np.exp(-(x**2) - y**2)
                if eta <= 0.0:
                    eps[i, j] = 2.0
                else:
                    eps[i, j] = 1.0
        return eps, a, b

    mg.apply_boundary = apply_boundary
    mg.construct_jump_conditions = construct_jump_conditions

    u = mg.solve()
    u_exact = np.zeros_like(X)
    u_exact[(X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2] = (np.exp(-(X**2) - Y**2))[
        (X - 0.5) ** 2 + (Y - 0.5) ** 2 <= r0**2
    ]
    G = mg.nghost
    print(f"Max error: {np.max(np.abs(u-u_exact)[G:-G, G:-G]):.2e}")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = ax.scatter(X, Y, u, c=u, cmap="coolwarm")
    # scatter = ax.scatter(X, Y, u_exact, c=u_exact, cmap="coolwarm")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(scatter, ax=ax, label="$\\phi$")
    plt.tight_layout()
    plt.show()


def example5():
    """Example 5 from XuDong Liu 2000"""
    nx, ny = 64, 64
    origin = [-1.0, -1.0]
    L = [2.0, 2.0]
    dx = L[0] / nx
    dy = L[1] / ny
    x = np.arange(origin[0] - 2.5 * dx, origin[0] + L[0] + 3 * dx, dx)
    y = np.arange(origin[1] - 2.5 * dy, origin[1] + L[1] + 3 * dx, dy)
    Y, X = np.meshgrid(y, x)

    r0 = 0.5
    g = np.zeros_like(X)
    f = lambda u: 0

    mg = Multigrid(f, g, origin=origin, L=L, n_phys=[nx, ny], levels=4, tol=1e-3)
    mg.surface = lambda x, y: x**2 + y**2 - r0**2
    mg.normal = lambda x, y, dx, dy: (
        x / np.sqrt(x**2 + y**2),
        y / np.sqrt(x**2 + y**2),
    )

    def apply_boundary(u):
        """Apply Dirichlet"""
        G = mg.nghost
        for i in np.arange(u.shape[0]):
            for j in np.arange(u.shape[1]):
                if i >= G and i < u.shape[0] - G and j >= G and j < u.shape[1] - G:
                    continue
                x = mg.origin[0] + (i - G + 0.5) * (mg.L[0] / (u.shape[0] - 2 * G))
                y = mg.origin[1] + (j - G + 0.5) * (mg.L[1] / (u.shape[1] - 2 * G))
                u[i, j] = 1 + np.log(2 * np.sqrt(x**2 + y**2))
        return u

    def construct_jump_conditions(u):
        """Reconstruct permittivity and jump conditions on coarse grid."""
        eps = np.ones_like(u)
        a = np.zeros_like(u)
        b = 2 * np.ones_like(u)
        return eps, a, b

    mg.apply_boundary = apply_boundary
    mg.construct_jump_conditions = construct_jump_conditions

    u = mg.solve()
    u_exact = np.ones_like(X)
    u_exact[X**2 + Y**2 > r0**2] = (
        1 + np.log(2 * np.sqrt(X**2 + Y**2))[X**2 + Y**2 > r0**2]
    )

    G = mg.nghost
    print(f"Max error: {np.max(np.abs(u-u_exact)[G:-G, G:-G]):.2e}")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = ax.scatter(X, Y, u, c=u, cmap="coolwarm")
    # scatter = ax.scatter(X, Y, u_exact, c=u_exact, cmap="coolwarm")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(scatter, ax=ax, label="$\\phi$")
    plt.tight_layout()
    plt.show()


# FIXME: This example is not working properly, same issue as example3
def example6():
    """Example 6 from XuDong Liu 2000"""
    nx, ny = 64, 64
    origin = [-1.0, -1.0]
    L = [2.0, 2.0]
    dx = L[0] / nx
    dy = L[1] / ny
    x = np.arange(origin[0] - 2.5 * dx, origin[0] + L[0] + 3 * dx, dx)
    y = np.arange(origin[1] - 2.5 * dy, origin[1] + L[1] + 3 * dx, dy)
    Y, X = np.meshgrid(y, x)

    r0 = 0.5
    g = np.zeros_like(X)
    f = lambda u: 0

    mg = Multigrid(f, g, origin=origin, L=L, n_phys=[nx, ny], levels=4, tol=1e-3)
    mg.surface = lambda x, y: x**2 + y**2 - r0**2
    mg.normal = lambda x, y, dx, dy: (
        x / np.sqrt(x**2 + y**2),
        y / np.sqrt(x**2 + y**2),
    )

    def apply_boundary(u):
        """Apply Dirichlet"""
        G = mg.nghost
        u[:G, :] = 0.0
        u[-G:, :] = 0.0
        u[:, :G] = 0.0
        u[:, -G:] = 0.0
        return u

    def construct_jump_conditions(u):
        """Reconstruct permittivity and jump conditions on coarse grid."""
        G = mg.nghost
        eps = np.ones_like(u)
        a = np.zeros_like(u)
        b = np.zeros_like(u)
        for i in np.arange(G, u.shape[0] - G):
            for j in np.arange(G, u.shape[1] - G):
                x = mg.origin[0] + (i - G + 0.5) * (mg.L[0] / (u.shape[0] - 2 * G))
                y = mg.origin[1] + (j - G + 0.5) * (mg.L[1] / (u.shape[1] - 2 * G))
                a[i, j] = -np.exp(x) * np.cos(y)
                b[i, j] = 2 * np.exp(x) * (y * np.sin(y) - x * np.cos(y))
        return eps, a, b

    mg.apply_boundary = apply_boundary
    mg.construct_jump_conditions = construct_jump_conditions

    u = mg.solve()
    u_exact = np.zeros_like(X)
    u_exact[X**2 + Y**2 <= r0**2] = (np.exp(X) * np.cos(Y))[X**2 + Y**2 <= r0**2]

    G = mg.nghost
    print(f"Max error: {np.max(np.abs(u-u_exact)[G:-G, G:-G]):.2e}")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = ax.scatter(X, Y, u, c=u, cmap="coolwarm")
    # scatter = ax.scatter(X, Y, u_exact, c=u_exact, cmap="coolwarm")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(scatter, ax=ax, label="$\\phi$")
    plt.tight_layout()
    plt.show()


# FIXME: This example is not working properly, same issue as example3
def example7():
    """Example 7 from XuDong Liu 2000"""
    nx, ny = 64, 64
    origin = [-1.0, -1.0]
    L = [2.0, 2.0]
    dx = L[0] / nx
    dy = L[1] / ny
    x = np.arange(origin[0] - 2.5 * dx, origin[0] + L[0] + 3 * dx, dx)
    y = np.arange(origin[1] - 2.5 * dy, origin[1] + L[1] + 3 * dx, dy)
    Y, X = np.meshgrid(y, x)

    r0 = 0.5
    g = np.zeros_like(X)
    f = lambda u: 0

    mg = Multigrid(f, g, origin=origin, L=L, n_phys=[nx, ny], levels=1, tol=1e-4)
    mg.surface = lambda x, y: x**2 + y**2 - r0**2
    mg.normal = lambda x, y, dx, dy: (
        x / np.sqrt(x**2 + y**2),
        y / np.sqrt(x**2 + y**2),
    )

    def apply_boundary(u):
        """Apply Dirichlet"""
        G = mg.nghost
        u[:G, :] = 0.0
        u[-G:, :] = 0.0
        u[:, :G] = 0.0
        u[:, -G:] = 0.0
        return u

    def construct_jump_conditions(u):
        """Reconstruct permittivity and jump conditions on coarse grid."""
        G = mg.nghost
        eps = np.ones_like(u)
        a = np.zeros_like(u)
        b = np.zeros_like(u)
        for i in np.arange(G, u.shape[0] - G):
            for j in np.arange(G, u.shape[1] - G):
                x = mg.origin[0] + (i - G + 0.5) * (mg.L[0] / (u.shape[0] - 2 * G))
                y = mg.origin[1] + (j - G + 0.5) * (mg.L[1] / (u.shape[1] - 2 * G))
                a[i, j] = y**2 - x**2
                b[i, j] = 4 * (y**2 - x**2)
        return eps, a, b

    mg.apply_boundary = apply_boundary
    mg.construct_jump_conditions = construct_jump_conditions

    u = mg.solve()
    u_exact = np.zeros_like(X)
    u_exact[X**2 + Y**2 <= r0**2] = (X**2 - Y**2)[X**2 + Y**2 <= r0**2]

    G = mg.nghost
    print(f"Max error: {np.max(np.abs(u-u_exact)[G:-G, G:-G]):.2e}")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    scatter = ax.scatter(X, Y, u, c=u, cmap="coolwarm")
    # scatter = ax.scatter(X, Y, u_exact, c=u_exact, cmap="coolwarm")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(scatter, ax=ax, label="$\\phi$")
    plt.tight_layout()
    plt.show()


# Using levels=1 the solution resembles the solution of the example 4 from Hongtao Liu 2023.
# Using levels > 1 the solution has some some extra features (drop in electric field) that are not present in the original example.
# Larger the levels, it seems larger the drop in electric field.
def example_cylinder():
    """Example 4 from Hongtao Liu"""
    nx, ny = 128, 64
    origin = [0.0, 0.0]
    L = [1.0, 0.5]
    dx = L[0] / nx
    dy = L[1] / ny
    x = np.arange(origin[0] - 2.5 * dx, origin[0] + L[0] + 3 * dx, dx)
    y = np.arange(origin[1] - 2.5 * dy, origin[1] + L[1] + 3 * dy, dy)
    Y, X = np.meshgrid(y, x)

    g = np.zeros_like(X)
    f = lambda u: -np.exp(u)
    mg = Multigrid(f, g, origin=origin, L=L, n_phys=[nx, ny], levels=1, tol=1e-4)
    mg.max_iter = 20
    r0 = 0.125
    mg.surface = lambda x, y: (x - 0.375) ** 2 + y**2 - r0**2
    mg.normal = lambda x, y, dx, dy: (
        (x - 0.375) / np.sqrt((x - 0.375) ** 2 + y**2),
        y / np.sqrt((x - 0.375) ** 2 + y**2),
    )

    def apply_boundary(u):
        G = mg.nghost
        for i in np.arange(G, u.shape[0] - G):
            for j in np.arange(G, u.shape[1] - G):
                x = mg.origin[0] + (i - G + 0.5) * (mg.L[0] / (u.shape[0] - 2 * G))
                y = mg.origin[1] + (j - G + 0.5) * (mg.L[1] / (u.shape[1] - 2 * G))
                if mg.surface(x, y) <= 0.0:
                    u[i, j] = -66.67
        for k in range(G):
            u[k, :] = 0.0
            u[-k - 1, :] = u[-G - 2, :]
            u[:, k] = u[:, G + 1]
            u[:, -k - 1] = u[:, -G - 2]
        return u

    def construct_jump_conditions(u):
        """Reconstruct permittivity and jump conditions on coarse grid."""
        eps = np.ones_like(u)
        a = np.zeros_like(u)
        b = np.zeros_like(u)
        # G = mg.nghost
        # for i in np.arange(G, u.shape[0] - G):
        #     for j in np.arange(G, u.shape[1] - G):
        #         x = mg.origin[0] + (i - G + 0.5) * (mg.L[0] / (u.shape[0] - 2 * G))
        #         y = mg.origin[1] + (j - G + 0.5) * (mg.L[1] / (u.shape[1] - 2 * G))
        #         if mg.surface(x, y) <= 0.0:
        #             eps[i, j] = 100
        return eps, a, b

    mg.apply_boundary = apply_boundary
    mg.construct_jump_conditions = construct_jump_conditions

    u = mg.solve(u=-50 * np.ones_like(X))

    plt.figure(figsize=(6, 3))
    plt.pcolormesh(X, Y, u, cmap="jet")
    plt.colorbar(label="$\\phi$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential Field")

    G = mg.nghost
    plt.figure(figsize=(6, 3))
    plt.plot(x, u[:, G], label="$y=0$")
    plt.plot(x, u[:, -G - 1], label="$y=0.5$")
    plt.xlabel("x")
    plt.ylabel("$\\phi$")
    plt.title("Potential Field along y=0 and y=0.5")

    Ex, Ey = np.gradient(-u, dx, dy)
    plt.figure(figsize=(6, 3))
    plt.plot(x, Ex[:, G], label="$y=0$")
    plt.plot(x, Ex[:, -G - 1], label="$y=0.5$")
    plt.xlabel("x")
    plt.ylabel("$E$")
    plt.title("Electric Field along y=0 and y=0.5")
    plt.show()


def example_nonlinear1():
    """Example 4 from Hongtao Liu"""
    nx, ny = 64, 64
    origin = [0.0, 0.0]
    L = [1.0, 1.0]
    dx = L[0] / nx
    dy = L[1] / ny
    x = np.arange(origin[0] - 2.5 * dx, origin[0] + L[0] + 3 * dx, dx)
    y = np.arange(origin[1] - 2.5 * dy, origin[1] + L[1] + 3 * dy, dy)
    Y, X = np.meshgrid(y, x)

    g = np.zeros_like(X)
    f = lambda u: -u
    mg = Multigrid(f, g, origin=origin, L=L, n_phys=[nx, ny], levels=3, tol=1e-12)
    mg.surface = lambda x, y: x
    mg.normal = lambda x, y, dx, dy: (1.0, 0.0)

    def apply_boundary(u):
        G = mg.nghost
        for k in range(G):
            u[k, :] = 1.0
            u[-k - 1, :] = np.exp(1.0)
            u[:, k] = u[:, G + 1]
            u[:, -k - 1] = u[:, -G - 2]
        return u

    def construct_jump_conditions(u):
        """Reconstruct permittivity and jump conditions on coarse grid."""
        eps = np.ones_like(u)
        a = np.zeros_like(u)
        b = np.zeros_like(u)
        return eps, a, b

    mg.apply_boundary = apply_boundary
    mg.construct_jump_conditions = construct_jump_conditions

    u = mg.solve()

    plt.figure()
    plt.pcolormesh(X, Y, u, cmap="jet")
    plt.colorbar(label="$\\phi$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential Field")

    G = mg.nghost
    u_exact = np.exp(X)
    plt.figure()
    plt.plot(x[G:-G], u[G:-G, G], label="Numerical")
    plt.plot(x[G:-G], u_exact[G:-G, G], "--", label="Exact")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$\\phi$")
    plt.title("Potential Field")
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # example1()
    # example3()
    # example5()
    # example6()
    # example7()
    # example_cylinder()
    example_nonlinear1()

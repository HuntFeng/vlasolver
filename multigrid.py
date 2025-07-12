import numpy as np


class MultigridCellCentered:
    """
    2D Cell-centered Multigrid solver for the nonlinear Poisson equation using Full Approximation Scheme (FAS):
        u_xx + u_yy + f(u) = g in (a, b)x(c, d),
        Dirichlet on left/right, Neumann (zero-gradient) on top/bottom.
        The grid includes 3 ghost cells on each side (cell-centered).
    """

    def __init__(
        self, f, g, L=[10.0, 10.0], n_phys=[64, 64], levels=4, tol=1e-6, max_iter=100
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
        self.n_phys = n_phys
        self.levels = levels
        self.tol = tol
        self.max_iter = max_iter
        # Total size including 3 ghost cells per side
        self.nghost = 3
        self.n = [n_phys[0] + 2 * self.nghost, n_phys[1] + 2 * self.nghost]
        self.u = np.zeros(self.n)

    def _apply_boundary(self, u):
        """Apply Dirichlet (left/right) and Neumann (top/bottom) BCs on 3 ghost cells."""
        G = self.nghost
        dx = self.L[0] / (u.shape[0] - 2 * G)
        dy = self.L[1] / (u.shape[1] - 2 * G)

        i_range = np.arange(G, u.shape[0] - G)
        j_range = np.arange(G, u.shape[1] - G)
        np.random.shuffle(i_range)
        np.random.shuffle(j_range)
        for i in i_range:
            for j in j_range:
                x = dx * (i - G + 0.5)
                y = dy * (j - G + 0.5)
                if (x - 0.375) ** 2 + y**2 - 0.125**2 < 0.0:
                    u[i, j] = -66.76

        # Dirichlet on left
        for k in range(G):
            u[k, :] = 0.0
        # Neumann on right
        for k in range(G):
            u[-(k + 1), :] = u[-(G + 2), :]
        # Neumann on bottom
        for k in range(G):
            u[:, k] = u[:, G + 1]
        # Neumann on top
        for k in range(G):
            u[:, -(k + 1)] = u[:, -(G + 2)]

        return u

    def _smooth_nonlin_gs(self, u, g, iter=3):
        """Nonlinear Gauss-Seidel smoothing with under-relaxation."""
        G = self.nghost
        dx = self.L[0] / (u.shape[0] - 2 * G)
        dy = self.L[1] / (u.shape[1] - 2 * G)
        omega = 1.0
        f = self.f
        for _ in range(iter):
            i_range = np.arange(G, u.shape[0] - G)
            j_range = np.arange(G, u.shape[1] - G)
            np.random.shuffle(i_range)
            np.random.shuffle(j_range)
            for i in i_range:
                for j in j_range:
                    if (i + j) % 2 == 0:
                        u_old = u[i, j]
                        denom = 2 / dx**2 + 2 / dy**2
                        avg = (u[i - 1, j] + u[i + 1, j]) / dx**2 + (
                            u[i, j - 1] + u[i, j + 1]
                        ) / dy**2
                        u_new = (avg - g[i, j] + f(u_old)) / denom
                        u[i, j] = (1 - omega) * u_old + omega * u_new

            np.random.shuffle(i_range)
            np.random.shuffle(j_range)
            for i in i_range:
                for j in j_range:
                    if (i + j) % 2 == 1:
                        u_old = u[i, j]
                        denom = 2 / dx**2 + 2 / dy**2
                        avg = (u[i - 1, j] + u[i + 1, j]) / dx**2 + (
                            u[i, j - 1] + u[i, j + 1]
                        ) / dy**2
                        u_new = (avg - g[i, j] + f(u_old)) / denom
                        u[i, j] = (1 - omega) * u_old + omega * u_new
            u = self._apply_boundary(u)
        return u

    def _nonlinear_operator(self, u):
        """N(u) = u_xx + u_yy + f(u), returns N(u)."""
        G = self.nghost
        dx = self.L[0] / (u.shape[0] - 2 * G)
        dy = self.L[1] / (u.shape[1] - 2 * G)
        out = np.zeros_like(u)
        f = self.f

        i_range = np.arange(G, u.shape[0] - G)
        j_range = np.arange(G, u.shape[1] - G)
        np.random.shuffle(i_range)
        np.random.shuffle(j_range)
        for i in i_range:
            for j in j_range:
                ddx = (u[i - 1, j] - 2 * u[i, j] + u[i + 1, j]) / dx**2
                ddy = (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1]) / dy**2
                laplace = ddx + ddy
                if (np.abs(ddx) > 1e6).any():
                    print(
                        f"ddx({i}, {j}) = {u[i - 1, j]} - 2 * {u[i, j]} + {u[i + 1, j]} / {dx}^2 = {ddx}"
                    )
                    raise ValueError("ddx has inf values.")
                if (np.abs(ddx) > 1e6).any():
                    raise ValueError("ddy has inf values.")
                out[i, j] = laplace + f(u[i, j])

        return out

    def _restrict(self, u_fine):
        """Restrict to coarse grid using full-weighting (cell-centered)."""
        G = self.nghost
        nf_x, nf_y = u_fine.shape
        nc_x = (nf_x - 2 * G) // 2 + 2 * G
        nc_y = (nf_y - 2 * G) // 2 + 2 * G
        u_coarse = np.zeros((nc_x, nc_y))
        # Loop over coarse grid (cell-centered)

        i_range = np.arange(G, nc_x - G)
        j_range = np.arange(G, nc_y - G)
        np.random.shuffle(i_range)
        np.random.shuffle(j_range)
        for ic in i_range:
            for jc in j_range:
                i_f = 2 * (ic - G) + G
                j_f = 2 * (jc - G) + G
                u_coarse[ic, jc] = 0.25 * (
                    u_fine[i_f, j_f]
                    + u_fine[i_f + 1, j_f]
                    + u_fine[i_f, j_f + 1]
                    + u_fine[i_f + 1, j_f + 1]
                )
        return u_coarse

    def _prolong(self, ec, n_fine):
        """Prolongate error to finer grid using bilinear interpolation (cell-centered)."""
        ef = np.zeros(n_fine)
        G = self.nghost
        nf_x, nf_y = n_fine

        i_range = np.arange(G, nf_x - G)
        j_range = np.arange(G, nf_y - G)
        np.random.shuffle(i_range)
        np.random.shuffle(j_range)
        for i in i_range:
            for j in j_range:
                ic = (i - G) // 2 + G
                jc = (j - G) // 2 + G
                ef[i, j] = ec[ic, jc]
                # if (i - G) % 2 == 0 and (j - G) % 2 == 0:
                #     ef[i, j] = ec[ic, jc]
                # elif (i - G) % 2 == 1 and (j - G) % 2 == 0:
                #     ef[i, j] = 0.5 * (ec[ic, jc] + ec[ic + 1, jc])
                # elif (i - G) % 2 == 0 and (j - G) % 2 == 1:
                #     ef[i, j] = 0.5 * (ec[ic, jc] + ec[ic, jc + 1])
                # else:
                #     ef[i, j] = 0.25 * (
                #         ec[ic, jc]
                #         + ec[ic + 1, jc]
                #         + ec[ic, jc + 1]
                #         + ec[ic + 1, jc + 1]
                # )
        return ef

    def _fas_v_cycle(self, level, u, g):
        """Full Approximation Scheme (FAS) V-cycle."""
        u = self._smooth_nonlin_gs(u, g, iter=10)
        if level == self.levels - 1:
            u = self._smooth_nonlin_gs(u, g, iter=20)
            return u
        lhs = self._nonlinear_operator(u)
        u_c = self._restrict(u)
        lhs_c = self._restrict(lhs)
        g_c = self._restrict(g)
        u_c = self._apply_boundary(u_c)
        u_c_old = u_c.copy()
        tau_c = lhs_c - self._nonlinear_operator(u_c)
        g_fas = g_c + tau_c
        u_c = self._fas_v_cycle(level + 1, u_c, g_fas)
        corr = self._prolong(u_c - u_c_old, u.shape)
        u += corr
        u = self._apply_boundary(u)
        u = self._smooth_nonlin_gs(u, g, iter=10)
        return u

    def solve(self):
        """Solve the nonlinear Poisson's equation with FAS multigrid (cell-centered, 3 ghost cells, Dirichlet/Neumann BCs)."""
        G = self.nghost
        u = np.zeros(self.n)
        g = np.zeros(self.n)
        g[G:-G, G:-G] = self.g
        u = self._apply_boundary(u)
        for i in range(self.max_iter):
            u_old = u.copy()
            u = self._fas_v_cycle(0, u, g)
            err = np.max(np.abs(u - u_old)[G:-G, G:-G])
            print(f"Iteration {i + 1}, max error: {err:.2e}")
            if err < self.tol:
                print(f"Converged after {i + 1} V-cycles with error {err:.2e}")
                break
        return u[G:-G, G:-G]


def compute_E_field(u, L, n_phys):
    """
    Compute the electric field from the potential u.
    Args:
        u: array, potential field
        L: list, physical domain size
        n_phys: list, number of physical grid points [Nx, Ny]
    Returns:
        Ex, Ey: arrays, electric field components
    """
    dx = L[0] / n_phys[0]
    dy = L[1] / n_phys[1]
    Ex, Ey = np.gradient(-u, dx, dy)
    return Ex, Ey


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example nonlinear Poisson:
    f = lambda u: -np.exp(u)
    # f = lambda u: 0
    n = 128  # Physical grid size (excluding ghost cells)
    dx = 1.0 / n
    dy = 0.5 / (n // 2)
    x = np.arange(dx / 2, 1.0, dx)
    y = np.arange(dy / 2, 0.5, dy)
    Y, X = np.meshgrid(y, x)
    # g = -100 * np.ones_like(X)
    g = np.zeros_like(X)
    mg = MultigridCellCentered(
        f, g, L=[1.0, 0.5], n_phys=[n, n // 2], levels=4, tol=1e-2
    )
    u = mg.solve()
    plt.figure(figsize=(6, 3))
    plt.pcolormesh(X, Y, u, cmap="jet")
    plt.colorbar(label="$\\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential Field")

    Ex, Ey = compute_E_field(u, mg.L, mg.n_phys)
    plt.figure(figsize=(6, 3))
    plt.pcolormesh(X, Y, Ex, cmap="jet")
    plt.colorbar(label="$E_x(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("$E_x$ Field")

    plt.figure()
    plt.plot(x, u[:, 0], label="$\\phi(x, 0)$")
    plt.plot(x, u[:, -1], label="$\\phi(x, 0.5)$")
    plt.xlabel("x")
    plt.ylabel("$\\phi$")

    plt.figure()
    plt.plot(x, Ex[:, 0], label="$E_x(x, 0)$")
    plt.plot(x, Ex[:, -1], label="$E_x(x, 0.5)$")
    plt.xlabel("x")
    plt.ylabel("$E_x$")
    plt.show()

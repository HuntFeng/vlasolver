import numpy as np


class GaussSeidel:
    def __init__(self, L=[1.0, 0.5]):
        self.max_iter = 5000
        self.L = L

    def apply_boundary(self, u):
        """Apply boundary conditions to the solution array."""
        dx = self.L[0] / (u.shape[0] - 2 * G)
        dy = self.L[1] / (u.shape[1] - 2 * G)
        for i in range(G, u.shape[0] - G):
            for j in range(G, u.shape[1] - G):
                x = dx * (i - G + 0.5)
                y = dy * (j - G + 0.5)
                if (x - 0.375) ** 2 + y**2 - 0.125**2 < 0.0:
                    u[i, j] = -66.76

        # Dirichlet condition on left side
        for k in range(G):
            u[k, :] = 0.0
        # Neumann condition on right side
        for k in range(G):
            u[-(k + 1), :] = u[-(G + 2), :]
        # Neumann condition on bottom side
        for k in range(G):
            u[:, k] = u[:, G + 1]
        # Neumann condition on top side
        for k in range(G):
            u[:, -(k + 1)] = u[:, -(G + 2)]
        return u

    def compute_error(self, u_old, u):
        """Compute the error between the current solution and the source term."""
        return np.abs(u - u_old)[G:-G, G:-G].max()

    def solve(self, u, b):
        """Perform Gauss-Seidel iteration."""
        omega = 1.5

        dx = self.L[0] / (u.shape[0] - 2 * G)
        dy = self.L[1] / (u.shape[1] - 2 * G)
        denom = 2 / dx**2 + 2 / dy**2
        for iter in range(self.max_iter):
            u_old = u.copy()
            for i in range(G, u.shape[0] - G):
                for j in range(G, u.shape[1] - G):
                    if (i + j) % 2 == 0:
                        avg = (u[i - 1, j] + u[i + 1, j]) / dx**2 + (
                            u[i, j - 1] + u[i, j + 1]
                        ) / dy**2
                        u_new = avg / denom
                        u[i, j] = (1 - omega) * u[i, j] + omega * u_new
                        # u[i, j] = (1 - omega) * u[i, j] + 0.25 * omega * (
                        #     u[i - 1, j]
                        #     + u[i + 1, j]
                        #     + u[i, j - 1]
                        #     + u[i, j + 1]
                        #     + b[i, j]
                        # )

            for i in range(G, u.shape[0] - G):
                for j in range(G, u.shape[1] - G):
                    if (i + j) % 2 == 1:
                        avg = (u[i - 1, j] + u[i + 1, j]) / dx**2 + (
                            u[i, j - 1] + u[i, j + 1]
                        ) / dy**2
                        u_new = avg / denom
                        u[i, j] = (1 - omega) * u[i, j] + omega * u_new
                        # u[i, j] = (1 - omega) * u[i, j] + 0.25 * omega * (
                        #     u[i - 1, j]
                        #     + u[i + 1, j]
                        #     + u[i, j - 1]
                        #     + u[i, j + 1]
                        #     + b[i, j]
                        # )

            u = self.apply_boundary(u)
            error = self.compute_error(u_old, u)
            print(f"Iteration {iter + 1}, max error: {error}")
            if error < 1e-2:
                print(f"Converged after {iter + 1} iterations.")
                break
        return u[G:-G, G:-G]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nx, ny = 166, 56
    G = 3  # Number of ghost cells
    gs = GaussSeidel(L=[1.0, 0.5])
    # u = np.zeros((nx, ny))
    u = -60 * np.ones((nx, ny))
    # b = -np.ones((nx, ny))
    b = np.zeros((nx, ny))
    u = gs.apply_boundary(u)
    u = gs.solve(u, b)

    dx = 1.0 / (nx - 2 * G)
    dy = 0.5 / (ny - 2 * G)
    x = np.arange(dx / 2, 1.0, dx)
    y = np.arange(dy / 2, 0.5, dy)
    Y, X = np.meshgrid(y, x)

    plt.figure(figsize=(6, 3))
    plt.pcolormesh(X, Y, u, cmap="jet")
    plt.colorbar(label="$\\phi(x,y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Potential Field")

    Ex, Ey = np.gradient(-u, dx, dy)
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

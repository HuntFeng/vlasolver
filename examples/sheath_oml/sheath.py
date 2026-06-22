import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

Te = 1.0  # electron temperature (normalized)
Ti = 0.1  # ion temperature (normalized)
mi = 2 * 1836.0  # ion mass (normalized)
me = 1.0  # electron mass (normalized)
vr = np.sqrt((Ti / Te) / (mi / me))
L = 20.0  # domain length (normalized)
x_wall = 2.5  # wall location (normalized); flat potential to its left
phi_w = -np.log(np.sqrt(mi / (2 * np.pi)))  # wall potential


def poisson_equation(x, y):
    # phi = y[0]
    phi = phi_w * np.exp(-(x - x_wall) / 2.5)  # Initial guess for potential
    dphi_dx = y[1]
    n_e = np.exp(phi)
    n_i = 1.0 / np.sqrt(1 - 2 * phi)
    d2phi_dx2 = -(n_i - n_e)
    return np.vstack((dphi_dx, d2phi_dx2))


# boundary conditions (both work):
# phi(x_wall) = phi_w, phi(L) = 0
def boundary_conditions(ya, yb):
    return np.array([ya[0] - phi_w, yb[0]])


dx = L / 125
x = np.arange(dx / 2, L, dx)

# solve the Poisson equation only to the right of the wall
mask = x >= x_wall
x_solve = x[mask]
y_guess = np.zeros((2, x_solve.size))
y_guess[0] = phi_w * np.exp(-(x_solve - x_wall) / 2.5)  # Initial guess for potential
solution = solve_bvp(
    poisson_equation, boundary_conditions, x_solve, y_guess, tol=1e-8, verbose=2
)
if solution.success:
    print("Solution converged successfully!")
else:
    print("Warning: Solution did not converge!")

# flat potential at phi_w to the left of the wall, solved profile to the right
phi = np.full(x.size, phi_w)
phi[mask] = solution.sol(x_solve)[0]
x /= L

# electron distribution function
# phi = y_guess[0]
plt.figure()
ve = np.linspace(-5, 5, 110)
X, V = np.meshgrid(x, ve, indexing="ij")
fe = np.exp(phi[:, None] - V**2 / 2) / np.sqrt(2 * np.pi)
v_ce = np.sqrt(2 * (phi - phi_w)[:, None] / me)  # cutoff velocity
fe[V > v_ce] = 0.0
plt.subplot(2, 1, 1)
plt.contourf(X, V, fe, levels=50, cmap="jet")
plt.colorbar(label="$f_e$")
plt.xlabel("$x$")
plt.ylabel("$v$")

# ion distribution function
vi = np.linspace(-15, 1, 110) * vr
X, V = np.meshgrid(x, vi, indexing="ij")
u0 = np.sqrt(Te / mi)
v_ci = -np.sqrt(2 * np.abs(phi)[:, None] / mi)  # cutoff velocity
fi = (
    np.exp(-((np.sqrt(V**2 - v_ci**2) - u0) ** 2) / (2 * vr**2))
    / np.sqrt(2 * np.pi)
    / vr
)
fi[V > v_ci] = 0.0
plt.subplot(2, 1, 2)
plt.contourf(X, V / vr, fi, levels=50, cmap="jet")
plt.colorbar(label="$f_i$")
plt.xlabel("$x$")
plt.ylabel("$v$")
plt.tight_layout()

plt.figure()
plt.subplot(2, 1, 1)
phi_guess = np.full(x.size, phi_w)
phi_guess[mask] = y_guess[0]
plt.plot(x, phi / (2 * Ti), label="solution")
plt.plot(x, phi_guess / (2 * Ti), "--", label="initial guess")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$\\phi$")
plt.axvline(x_wall / L, color="black", linestyle="--")
print(f"wall potential {phi_w / (2 * Ti)}")


ne = np.exp(phi)
ni = 1.0 / np.sqrt(1 - 2 * phi)
plt.subplot(2, 1, 2)
plt.plot(x, ne, label="$n_e$")
plt.plot(x, ni, label="$n_i$")
plt.axvline(x_wall / L, color="black", linestyle="--")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$n$")
plt.tight_layout()

plt.show()


# save initial potential to csv file
G = 3
phi_padded = np.pad(phi, (G, G), "edge")
np.savetxt(f"{os.path.dirname(__file__)}/potential.csv", phi_padded.T)

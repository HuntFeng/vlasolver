import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

Te = 1.0  # electron temperature (normalized)
Ti = 0.1  # ion temperature (normalized)
mi = 1836.0  # ion mass (normalized)
me = 1.0  # electron mass (normalized)
vr = np.sqrt((Ti / Te) / (mi / me))
L = 20.0  # domain length (normalized)
phi_w = -np.log(np.sqrt(mi / (2 * np.pi)))  # wall potential


def poisson_equation(x, y):
    # phi = y[0]
    phi = phi_w * np.exp(-x / 2.5)  # Initial guess for potential
    dphi_dx = y[1]
    n_e = np.exp(phi)
    n_i = 1.0 / np.sqrt(1 - 2 * phi)
    d2phi_dx2 = -(n_i - n_e)
    return np.vstack((dphi_dx, d2phi_dx2))


# boundary conditions (both work):
# phi(0) = phi_w, phi(L) = 0
# phi(0) = phi_w, phi'(L) = 0
def boundary_conditions(ya, yb):
    return np.array([ya[0] - phi_w, yb[0]])
    # return np.array([ya[0] - phi_w, yb[1]])


x = np.linspace(0, L, 100)
y_guess = np.zeros((2, x.size))
y_guess[0] = phi_w * np.exp(-x / 2.5)  # Initial guess for potential
solution = solve_bvp(
    poisson_equation, boundary_conditions, x, y_guess, tol=1e-8, verbose=2
)
if solution.success:
    print("Solution converged successfully!")
else:
    print("Warning: Solution did not converge!")

phi = solution.sol(x)[0]
np.savetxt("initial_potential.csv", np.vstack((x, phi)).T)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, phi, label="solution")
plt.plot(x, y_guess[0], "--", label="initial guess")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$\\phi$")

plt.subplot(2, 1, 2)
plt.plot(x, np.exp(phi), label="$n_e$")
plt.plot(x, 1.0 / np.sqrt(1 - 2 * phi), label="$n_i$")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$n$")
plt.tight_layout()

# electron distribution function
phi = y_guess[0]
plt.figure()
v = np.linspace(-5, 5, 100)
X, V = np.meshgrid(x, v, indexing="ij")
fe = np.exp(phi[:, None] - V**2 / 2) / np.sqrt(2 * np.pi)
v_ce = np.sqrt(2 * (phi - phi_w)[:, None] / me)  # cutoff velocity
fe[V > v_ce] = 0.0
plt.subplot(2, 1, 1)
plt.contourf(X, V, fe, levels=50, cmap="jet")
plt.colorbar(label="$f_e$")
plt.xlabel("$x$")
plt.ylabel("$v$")

# ion distribution function
v = np.linspace(-8, 1, 100) * vr
X, V = np.meshgrid(x, v, indexing="ij")
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
plt.show()

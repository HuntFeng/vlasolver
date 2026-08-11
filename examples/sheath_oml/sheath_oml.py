"""
2D cylindrical OML sheath around an infinitely long dust grain / collector.

Normalization: x = r/lambda_De, phi = e*Phi/(k_B*T_e),
lambda_De = sqrt(eps0*k_B*T_e/(n0*e**2)).
Poisson equation: phi'' + phi'/x = n_e/n0 - n_i/n0.

Densities follow the 2D2V cylindrical OML construction for a collisionless
Maxwellian plasma (monotonic potential, surface is the orbital barrier).

Density formulas are found in:
The orbital-motion-limited regime of cylindrical Langmuir probes
DOI 10.1063/1.873293.
Orbital motion theory and operational regimes for cylindrical emissive probes
DOI: 10.1063/1.4975088
"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_bvp
from scipy.optimize import fsolve
from scipy.special import roots_legendre, erf, k0, k1


def floating_potential(mr: float, Tr: float):
    """Normalized floating potential phi_d from ion/electron current balance.

    sqrt(mr/Tr) * exp(phi_d) + phi_d/Tr = 1.
    """
    solution = fsolve(lambda phi: np.sqrt(mr / Tr) * np.exp(phi) + phi / Tr - 1, -1.0)
    return solution[0]


@dataclass(frozen=True)
class OMLParameters:
    """Parameters for the normalized cylindrical OML problem."""

    a: float = 1.0  # dust radius / lambda_De
    rmax: float = 10.0  # outer boundary / lambda_De

    Tr: float = 1.0
    mr: float = 100.0


def _gauss_legendre_half_line(order, u_max):
    """Gauss-Legendre nodes/weights on [0, u_max]."""
    xi, wi = roots_legendre(order)
    u = 0.5 * u_max * (xi + 1.0)
    w = 0.5 * u_max * wi
    return u, w


def oml_density(
    r,
    phi,
    *,
    a,
    phi_d,
    Tr,
    q,
    quadrature_order=96,
    u_max=8.0,
):
    """n/n_infinity for a 2D cylindrical OML species (charge_number: +1 ions, -1 electrons).

    psi = q * phi / T_over_Te is the species-normalized potential.
    Density = incoming half-Maxwellian + particles reflected by the orbital
    barrier, from angular-momentum conservation in 2D2V velocity space.
    """
    psi = q * phi / Tr
    psi_d = q * phi_d / Tr

    # Incoming: u_r^2 >= max(-psi, 0) required for real energy at infinity.
    n_in = 0.5 * np.exp(-np.maximum(psi, 0.0))

    # Reflected: angular-momentum conservation bounds u_r between lower/upper
    # (z = r/a); lower also enforces non-negative energy at infinity.
    u_theta, weights = _gauss_legendre_half_line(quadrature_order, u_max)

    z2_minus_1 = (r / a) ** 2 - 1.0

    # Shape: (Nr, Nq)
    ut2 = u_theta[None, :] ** 2
    psi_col = psi[:, None]

    upper2 = z2_minus_1[:, None] * ut2 + psi_d - psi_col
    lower2 = np.maximum(0.0, -psi_col - ut2)

    valid = upper2 > lower2

    lower = np.sqrt(lower2)
    upper = np.sqrt(np.maximum(upper2, 0.0))

    # int exp(-u_r^2) du_r from lower to upper = sqrt(pi)/2 * [erf(upper)-erf(lower)].
    erf_difference = (erf(upper) - erf(lower)) * valid

    integrand = np.exp(-psi_col) * np.exp(-ut2) * erf_difference

    n_reflected = np.sum(integrand * weights[None, :], axis=1) / np.sqrt(np.pi)

    return n_in + n_reflected


def solve_oml(params=None):
    """Solve the normalized cylindrical OML-Poisson BVP.

    BCs: phi(a) = phi_d at the dust surface; linearized 2D Debye screening
    phi'(R) = -K1(R)/K0(R) * phi(R) at the outer boundary.

    Returns a BVPResult with sol.params and sol.density(r, phi) attached.
    """
    if params is None:
        params = OMLParameters()

    a = params.a
    R = params.rmax

    x = np.linspace(a, R, 250)

    phi_d = floating_potential(params.mr, params.Tr)
    guess = phi_d * k0(x) / k0(a)

    dguess = np.gradient(guess, x)
    y0 = np.vstack((guess, dguess))

    def density(x_local, phi_local):
        ni = oml_density(
            x_local,
            phi_local,
            a=a,
            Tr=params.Tr,
            q=1,
            phi_d=phi_d,
        )
        ne = oml_density(
            x_local,
            phi_local,
            a=a,
            Tr=1.0,
            q=-1,
            phi_d=phi_d,
        )
        return ni, ne

    def ode(x_local, y):
        phi = y[0]
        dphi = y[1]

        ni, ne = density(x_local, phi)

        # phi'' + phi'/x = ne - ni
        d2phi = ne - ni - dphi / x_local

        return np.vstack((dphi, d2phi))

    alpha = k1(R) / k0(R)  # dK0/dx = -K1

    def bc(ya, yb):
        return np.array([
            ya[0] - phi_d,
            yb[1] + alpha * yb[0],
        ])

    sol = solve_bvp(
        ode,
        bc,
        x,
        y0,
        tol=1e-6,
        max_nodes=5000,
        verbose=0,
    )

    if not sol.success:
        raise RuntimeError(
            "Cylindrical OML BVP did not converge: " + sol.message
        )

    sol.params = params

    def density_at(r, phi=None):
        r = np.asarray(r, dtype=float)
        if phi is None:
            phi = sol.sol(r)[0]
        ni, ne = density(r, np.asarray(phi, dtype=float))
        return ni, ne

    sol.density = density_at
    return sol


def plot_potential(sol, ax=None, show=True, label=None):
    """Plot the normalized potential phi = e*Phi/(k_B*T_e) from solve_oml()."""
    if ax is None:
        _, ax = plt.subplots()

    x = np.linspace(
        sol.params.a_over_lambda_D,
        sol.params.rmax_over_lambda_D,
        600,
    )
    phi = sol.sol(x)[0]

    if label is None:
        label = (
            rf"$\phi_d={sol.params.phi_d:g}$, "
            rf"$a/\lambda_{{De}}={sol.params.a_over_lambda_D:g}$"
        )

    ax.plot(x, phi, label=label)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$r/\lambda_{De}$")
    ax.set_ylabel(r"$e\Phi/(k_B T_e)$")
    ax.set_title("2D cylindrical OML potential")
    ax.grid(True, alpha=0.25)
    ax.legend()

    if show:
        plt.show()

    return ax


if __name__ == "__main__":
    params = OMLParameters(
        a=1.0,
        rmax=20.0,
        Tr=1.0,
        mr=100.0,
    )

    sol = solve_oml(params)
    print(f"Converged: {sol.success}")
    print(f"Number of BVP nodes: {sol.x.size}")
    print(f"phi(a) = {sol.sol(params.a)[0]:.8f}")
    print(f"phi(R) = {sol.sol(params.rmax)[0]:.8f}")

    plot_potential(sol)

"""
Demo of the parametrized MPP flux limiter (Xiong et al. 2014) applied to
the 1D linear advection equation, WITHOUT a CFL restriction, for any sign of a.

Large time-step flux follows Filbet et al. (2001) and Xiong et al. (2014)
eqs. (2.14)-(2.17): instead of a np.roll() grid shift, the integer-cell
contribution is accumulated as an explicit sum over swept cell averages,
exactly as written in those papers.

For a > 0, xi = a*dt/dx >= 1:
  Let i' be the index such that x*_{i+1/2} = x_{i+1/2} - a*dt lies in
  (x_{i'-1/2}, x_{i'+1/2}], i.e. i' = i - m  (mod N),  m = floor(xi).
  1st-order flux (eq. 2.14):
    h_{i+1/2} = sum_{j=i'+1}^{i} dx*u_j  +  (x_{i+1/2} - x*_{i+1/2})*u_{i'}
              = sum_{j=i'+1}^{i} u_j      +  alpha * u_{i'}        [in units of dx]
  High-order flux (eq. 2.15):
    H_{i+1/2} = sum_{j=i'+1}^{i} u_j      +  H~_{i'+1/2}
  where H~_{i'+1/2} is the standard (|xi|<1) PFC flux evaluated at i'.

For a < 0, xi = a*dt/dx <= -1:  mirror of the above (eqs. 2.16-2.17).
"""

import matplotlib.pyplot as plt
import numpy as np

# ── 1. Grid and initial condition ─────────────────────────────────────────────

N = 200
L = 2 * np.pi
dx = L / N
x = np.linspace(dx / 2, L - dx / 2, N)

u0 = np.sin(4.3 * x) ** 2
u_min = 0.0
u_max = u0.max()

# ── 2. Scheme parameters ──────────────────────────────────────────────────────

a = 1.0  # advection speed; code handles any sign
CFL = 2.5  # free to exceed 1
dt = CFL * dx / abs(a)
xi = a * dt / dx  # signed Courant number
T_end = 2 * np.pi

print(f"a = {a},  CFL = {CFL},  xi = {xi:.4f}")

# ── 3. Decompose xi ───────────────────────────────────────────────────────────


def decompose(xi):
    """
    Returns (m, alpha):
      m     = floor(|xi|)  >= 0  integer number of fully-swept cells
      alpha = |xi| - m     in [0, 1)  fractional Courant number
    """
    # m = int(np.floor(abs(xi)))
    m = int(abs(xi))
    alpha = abs(xi) - m
    return m, alpha


# ── 4. Sub-cell (fractional) PFC flux for |xi_frac| < 1 ──────────────────────
#    These implement the standard |alpha|<1 formulae from Filbet (2001) /
#    Xiong (2014) Appendix, used both standalone (small CFL) and as the
#    fractional piece H~ inside the large-CFL flux.


def _frac_first_order(u, xi_frac):
    """
    1st-order upwind flux at every face i+1/2, |xi_frac| < 1.
      xi_frac > 0 (a>0):  f_{i+1/2} =  alpha * u_i
      xi_frac < 0 (a<0):  f_{i+1/2} = -alpha * u_{i+1}
    """
    if xi_frac >= 0:
        return xi_frac * u
    else:
        alpha = -xi_frac
        return -alpha * np.roll(u, -1)


def _frac_third_order(u, xi_frac):
    """
    3rd-order PFC flux at every face i+1/2, |xi_frac| < 1.
    Xiong (2014) eq. (A.1)-(A.5) for a>0, (A.8)-(A.11) for a<0.
    """
    n = len(u)
    F = np.zeros(n)
    if xi_frac >= 0:
        alpha = xi_frac
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            val = (
                u[i]
                + (1 / 6) * (2 - alpha) * (1 - alpha) * (u[ip] - u[i])
                + (1 / 6) * (1 - alpha) * (1 + alpha) * (u[i] - u[im])
            )
            F[i] = alpha * val
    else:
        alpha = -xi_frac  # positive magnitude
        for i in range(n):
            ip = (i + 1) % n
            ipp = (i + 2) % n
            val = (
                u[ip]
                + (1 / 6) * (2 - alpha) * (1 - alpha) * (u[i] - u[ip])
                + (1 / 6) * (1 - alpha) * (1 + alpha) * (u[ip] - u[ipp])
            )
            F[i] = -alpha * val
    return F


# ── 5. Large-CFL flux: integer sum + fractional piece (Filbet / Xiong) ────────


def compute_fluxes(u, xi):
    """
    Compute first-order (f_lo) and high-order (F_hi) fluxes at all faces i+1/2
    for any value of xi (signed Courant number), following Filbet (2001) and
    Xiong (2014) eqs. (2.14)-(2.17).

    For a > 0 (xi = m + alpha >= 0), face i+1/2:
      The foot of the characteristic starting at x_{i+1/2} at t^{n+1} lands
      at x*_{i+1/2} = x_{i+1/2} - a*dt, which is in cell i' = i - m.

      h_{i+1/2} = [sum_{j=i-m+1}^{i} u_j]  +  alpha * u_{i-m}   (eq. 2.14)
      H_{i+1/2} = [sum_{j=i-m+1}^{i} u_j]  +  H~_{(i-m)+1/2}   (eq. 2.15)

    For a < 0 (xi = -(m + alpha) <= 0), face i+1/2:
      x*_{i+1/2} lands in cell i' = i + m + 1.

      h_{i+1/2} = -[sum_{j=i+1}^{i+m} u_j]  +  (-alpha) * u_{i+m+1}  (eq. 2.16)
      H_{i+1/2} = -[sum_{j=i+1}^{i+m} u_j]  +  H~_{(i+m)+1/2}        (eq. 2.17)

    When m = 0 this reduces exactly to the standard sub-cell formulae.
    All indices are taken mod N for periodic boundaries.
    """
    n = len(u)
    m, alpha = decompose(xi)
    xi_frac = alpha if xi >= 0 else -alpha  # signed fractional Courant

    # ── fractional PFC fluxes at every shifted index ──────────────────────────
    # For a>0 we need H~_{i'+1/2} where i' = i - m, i.e. shift the stencil
    # left by m.  Equivalently: roll u left by m, compute flux, result[i]
    # is H~_{(i-m)+1/2}.
    # For a<0 shift right by m.
    shift = m if xi >= 0 else -m
    u_shift = np.roll(u, shift)  # u_shift[i] = u[(i+shift) % n]

    F_frac_hi = _frac_third_order(u_shift, xi_frac)  # H~_{i'+1/2}
    F_frac_lo = _frac_first_order(u_shift, xi_frac)  # h~_{i'+1/2}

    # ── integer-cell sum (Filbet / Xiong convention) ───────────────────────────
    # For a > 0: integer_sum[i] = sum_{j=i-m+1}^{i} u_j
    #            = u[i] + u[i-1] + ... + u[i-m+1]  (m terms, 0 if m=0)
    # For a < 0: integer_sum[i] = -sum_{j=i+1}^{i+m} u_j
    #            = -(u[i+1] + ... + u[i+m])         (m terms, 0 if m=0)
    int_sum = np.zeros(n)
    if xi >= 0:
        # for k in range(1, m + 1):  # k = 1 .. m
        #     int_sum += np.roll(
        #         u, k
        #     )  # u_{i-k+... } accumulates u_{i}, u_{i-1}, ..., u_{i-m+1}
        #     # roll(u, k)[i] = u[i-k], so k=1 gives u[i], wait — need careful indexing:
        #     # We want sum_{j=i-m+1}^{i} u_j = u[i-(m-1)] + ... + u[i]
        #     # roll(u, k)[i] = u[(i-k) % n], so k=0..m-1 gives u[i], u[i-1], ..., u[i-m+1]
        # # redo with correct offset
        # int_sum = np.zeros(n)
        # for k in range(m):  # k = 0 .. m-1  → u[i-k]
        #     int_sum += np.roll(u, -k) if False else u  # placeholder — see below
        # cleaner: build directly
        int_sum = np.zeros(n)
        for k in range(m):  # want u[i], u[i-1], ..., u[i-(m-1)]
            int_sum += np.roll(u, k)  # roll(u, k)[i] = u[i-k] ✓
    else:
        for k in range(1, m + 1):  # want -(u[i+1] + ... + u[i+m])
            int_sum -= np.roll(u, -k)  # roll(u,-k)[i] = u[i+k] ✓

    f_lo = int_sum + F_frac_lo
    F_hi = int_sum + F_frac_hi
    return F_hi, f_lo


# ── 6. MPP limiter (Xiong 2014) ───────────────────────────────────────────────


def mpp_limiter(u, F_hi, f_lo):
    n = len(u)
    eps = 1e-13
    d = F_hi - f_lo
    # delta = u + np.roll(f_lo, 1) - f_lo
    delta = 0.0 - (u + np.roll(f_lo, 1) - f_lo)

    eps_l = np.ones(n)
    eps_r = np.ones(n)

    for i in range(n):
        di_m = d[i - 1]
        di_p = d[i]
        delt = delta[i]
        p = di_m - di_p - delt

        if di_m >= 0 and di_p < 0:
            eps_l[i] = 1.0
            eps_r[i] = 1.0
        elif di_m >= 0 and di_p > 0:
            eps_l[i] = 1.0
            eps_r[i] = min(1.0, delt / (-di_p))
        elif di_m < 0 and di_p <= 0:
            eps_l[i] = min(1.0, delt / (di_m))
            eps_r[i] = 1.0
        elif di_m < 0 and di_p > 0:
            if p > 0:
                eps_l[i] = 1.0
                eps_r[i] = 1.0
            else:
                eps_l[i] = min(1.0, delt / (di_m - di_p))
                eps_r[i] = min(1.0, delt / (di_m - di_p))

        # if di_m < 0 and di_p <= 0:
        #     eps_l[i] = min(1.0, delt / (di_m - eps))
        # elif di_m * di_p <= 0 and p < 0:
        #     eps_l[i] = delt / (di_m - di_p)
        # else:
        #     eps_l[i] = 1.0
        #
        # if di_m > 0 and di_p >= 0:
        #     eps_r[i] = min(1.0, -delt / (di_p + eps))
        # elif di_m * di_p <= 0 and p < 0:
        #     eps_r[i] = delt / (di_m - di_p)
        # else:
        #     eps_r[i] = 1.0

    eps_l = np.clip(eps_l, 0.0, 1.0)
    eps_r = np.clip(eps_r, 0.0, 1.0)

    theta = np.minimum(eps_r, np.roll(eps_l, -1))
    return theta


# ── 7. One time step ──────────────────────────────────────────────────────────


def step(u, use_limiter, xi):
    F_hi, f_lo = compute_fluxes(u, xi)

    if use_limiter:
        theta = mpp_limiter(u, F_hi, f_lo)
        F_use = theta * (F_hi - f_lo) + f_lo
    else:
        F_use = F_hi

    return u + np.roll(F_use, 1) - F_use


# ── 8. Time loop ──────────────────────────────────────────────────────────────


def run(u_init, use_limiter, T, xi_full):
    u = u_init.copy()
    t = 0.0
    neg = []
    while t < T - 1e-12:
        dt_now = min(dt, T - t)
        xi_now = a * dt_now / dx
        u = step(u, use_limiter, xi_now)
        t += dt_now
        neg.append(u.min())
    return u, neg


u_no_lim, neg_no = run(u0, use_limiter=False, T=T_end, xi_full=xi)
u_with_lim, neg_wi = run(u0, use_limiter=True, T=T_end, xi_full=xi)

# ── 9. Plot ───────────────────────────────────────────────────────────────────

m, alpha = decompose(xi)
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
fig.suptitle(
    f"a = {a},  CFL = {CFL},  xi = {xi:.2f}  "
    f"(integer shifts m = {m}, alpha = {alpha:.2f})  "
    f"[Filbet/Xiong sum convention]"
)

ax = axes[0]
ax.plot(x, u0, "k--", lw=1.5, label="Exact (=initial)")
ax.plot(x, u_no_lim, "r-o", ms=4, label="3rd-order, no limiter")
ax.plot(x, u_with_lim, "b-s", ms=4, label="3rd-order + MPP limiter")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.set_title("(a) Solution after one full period")
ax.set_xlabel("x")
ax.legend(fontsize=8)
ax.set_ylim(-0.15, 1.1)

ax = axes[1]
ax.plot(neg_no, "r-", label="No limiter")
ax.plot(neg_wi, "b-", label="With MPP limiter")
ax.axhline(0, color="gray", lw=0.8, ls=":", label="Physical min = 0")
ax.set_title("(b) Minimum cell value vs. time step")
ax.set_xlabel("Time step")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("mpp_limiter_filbet_sum.png", dpi=150)
plt.show()

# ── 10. Summary ───────────────────────────────────────────────────────────────

print("=" * 55)
print(f"xi = {xi:.4f}  =>  m = {m},  alpha = {alpha:.4f}")
print(f"Exact bounds:          [{u_min:.6f},  {u_max:.6f}]")
print("-" * 55)
print(f"No limiter  - min: {u_no_lim.min():.6f}   max: {u_no_lim.max():.6f}")
print(f"With limiter- min: {u_with_lim.min():.6f}   max: {u_with_lim.max():.6f}")
print("-" * 55)
print(f"Negative cells without limiter: {(u_no_lim  < -1e-12).sum()}")
print(f"Negative cells with    limiter: {(u_with_lim < -1e-12).sum()}")
print(f"L1 error (no lim):   {np.mean(np.abs(u_no_lim  - u0)):.4e}")
print(f"L1 error (with lim): {np.mean(np.abs(u_with_lim - u0)):.4e}")

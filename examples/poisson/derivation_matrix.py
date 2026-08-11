"""
Derivation of equations (30) and (33) from:
  Cho et al., "A Second-Order Boundary Condition Capturing Method for Solving
  the Elliptic Interface Problems on Irregular Domains", J Sci Comput (2019).

Notation (paper convention, assuming (i,j) in Omega-):
  - xR = (x_i + theta_R * dx, y_j)    interface or grid point to the right
  - xL = (x_i - theta_L * dx, y_j)    interface or grid point to the left
  - xT = (x_i, y_j + theta_T * dy)    interface or grid point to the top
  - xB = (x_i, y_j - theta_B * dy)    interface or grid point to the bottom

  - u-R, u+L, u+T, u-B: approximated values of u at the interface points
    in Omega- (superscript minus) or Omega+ (superscript plus).

  - a = [u] (jump in solution), b = [beta * u_n] (jump in normal flux)
  - beta+, beta-, [beta] = beta+ - beta-

Sec 3.2.2:
  Eq.(29) — second-order finite difference approximations of u_x at interface.
  Eq.(30) — algebraic expression for [beta u_x] at xR (Case 1).
  Eq.(32) — second-order FD approximations for Case 2 (extra point xr).
  Eq.(33) — algebraic expression for [beta u_x] at xR and xr (Case 2).

This script derives both equations for all 4 Cartesian directions
(Right, Left, Top, Bottom).

IMPORTANT CONVENTIONS (getting these wrong caused real implementation bugs):

1. The SECOND interface point theta_{D2} is measured from the FAR grid point
   BACK toward the cell (paper eq. 31):
       x_r = x_{i+2} - theta_{R2} * dx     (Right direction)
   NOT forward from x_{i+1}. In the solver this means
       theta_rr = compute_theta(Direction::L, i + 2, j)
   and NOT compute_theta(Direction::R, i + 1, j).

2. theta_other is the theta of the OPPOSITE direction on the minus side
   (e.g. theta_L for the Right equation). It is 1 when that side is uncut
   (the far stencil point is then the grid point u_{i-1}). When the opposite
   side IS cut (case 4: three cut directions always contain an opposite
   pair), keep theta_other < 1: the far stencil value becomes the other
   interface unknown u^-_{opp}, i.e. an OFF-DIAGONAL entry of B, and the
   u_{i-1} column of C must be dropped. The coefficients below (c_far, c_uc)
   are derived for general theta_other precisely for this purpose.

3. In eq.(33) BOTH jump values a (at x_D) and a_r (at x_{D2}) appear in BOTH
   rows (cross terms), because u^+ at both collinear interfaces enters each
   one-sided Omega+ stencil. Note a_vec and a2_vec below each have two
   nonzero entries; the paper's eq.(33) displays only the diagonal terms.

4. Only u^-_D (the first interface value) is part of the quadratic
   polynomial P stencil used for the geometric jump approximation
   (eqs. 26-28). The second interface value u^-_{D2} is NOT, so when
   assembling M no grad_coeff term may be subtracted from its column.
"""

import sympy as sp
from sympy import init_printing
from IPython.display import display, Markdown, Math

init_printing()

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
dx, dy = sp.symbols("Delta_x Delta_y")

th_R, th_L, th_T, th_B = sp.symbols("theta_R theta_L theta_T theta_B")
th_R2, th_L2, th_T2, th_B2 = sp.symbols("theta_{R2} theta_{L2} theta_{T2} theta_{B2}")

bp = sp.Symbol("beta^+")
bm = sp.Symbol("beta^-")
bj = sp.Symbol("[beta]")
a = sp.Symbol("a")
b = sp.Symbol("b")
a2 = sp.Symbol("a_r")
b2 = sp.Symbol("b_r")

uc = sp.Symbol("u_{i,j}")
u_p1 = sp.Symbol("u_{i+1,j}")
u_p2 = sp.Symbol("u_{i+2,j}")
u_p3 = sp.Symbol("u_{i+3,j}")
u_m1 = sp.Symbol("u_{i-1,j}")
u_m2 = sp.Symbol("u_{i-2,j}")
u_m3 = sp.Symbol("u_{i-3,j}")
u_t1 = sp.Symbol("u_{i,j+1}")
u_t2 = sp.Symbol("u_{i,j+2}")
u_t3 = sp.Symbol("u_{i,j+3}")
u_b1 = sp.Symbol("u_{i,j-1}")
u_b2 = sp.Symbol("u_{i,j-2}")
u_b3 = sp.Symbol("u_{i,j-3}")

uRm = sp.Symbol("u_R^-")
uLm = sp.Symbol("u_L^-")
uTm = sp.Symbol("u_T^-")
uBm = sp.Symbol("u_B^-")
uRm2 = sp.Symbol("u_{R2}^-")
uLm2 = sp.Symbol("u_{L2}^-")
uTm2 = sp.Symbol("u_{T2}^-")
uBm2 = sp.Symbol("u_{B2}^-")


# ===================================================================
# Helper: 3-point FD coefficients
# ===================================================================
def fd_coeffs_3pt(x0, x1, x2, x_eval):
    """Return (c0,c1,c2) such that f'(x_eval) ≈ c0*f(x0) + c1*f(x1) + c2*f(x2)."""
    c0 = (2 * x_eval - x1 - x2) / ((x0 - x1) * (x0 - x2))
    c1 = (2 * x_eval - x0 - x2) / ((x1 - x0) * (x1 - x2))
    c2 = (2 * x_eval - x0 - x1) / ((x2 - x0) * (x2 - x1))
    return sp.simplify(c0), sp.simplify(c1), sp.simplify(c2)


# ===================================================================
# SECTION 1 — Equation (30) for all 4 directions (Case 1)
# ===================================================================
display(Markdown("## Equation (30) — Case 1: single interface point per direction"))


def derive_eq30(
    direction_name,
    th_self,
    th_other,
    u_self_m,
    u_far_m,
    u_far_grid,
    u_near_grid,
    u_far2_grid,
    h,
):
    """
    Derive Eq.(30) for a single direction.

    The algebraic approximation of [beta u_n] at the interface:

        [beta+ u+_n - beta- u-_n] ≈ [B · (u⁻_D, u⁻_far) + C · (uc, u_near, u_far2)
                                     + a_coeff · a] · h

    Returns (B_vec, C_vec, a_coeff, expr).
    """
    th = th_self
    to = th_other

    # Omega+ side: stencil through {u_interface_p, u_near, u_far2}
    # Local coords (interface at 0):
    #   x_R = 0,  x_{near} = (1-th)*h,  x_{far2} = (2-th)*h
    x0_p, x1_p, x2_p = 0, (1 - th) * h, (2 - th) * h
    cp0, cp1, cp2 = fd_coeffs_3pt(x0_p, x1_p, x2_p, x0_p)

    # Omega- side: stencil through {u_far_m, uc, u_self_m}
    #   x_far = -(th+to)*h,  x_i = -th*h,  x_self = 0
    x0_m, x1_m, x2_m = -(th + to) * h, -th * h, 0
    cm0, cm1, cm2 = fd_coeffs_3pt(x0_m, x1_m, x2_m, x2_m)

    # Build [beta u_n] = beta+ * u+_n - beta- * u-_n
    u_self_p = u_self_m + a
    ux_plus = cp0 * u_self_p + cp1 * u_near_grid + cp2 * u_far2_grid
    ux_minus = cm0 * u_far_m + cm1 * uc + cm2 * u_self_m
    jump_algebraic = sp.simplify(bp * ux_plus - bm * ux_minus)
    expr_h = sp.expand(sp.simplify(jump_algebraic * h))

    c_self = sp.simplify(expr_h.coeff(u_self_m))
    c_far = sp.simplify(expr_h.coeff(u_far_m))
    c_uc = sp.simplify(expr_h.coeff(uc))
    c_near = sp.simplify(expr_h.coeff(u_near_grid))
    c_far2 = sp.simplify(expr_h.coeff(u_far2_grid))
    c_a = sp.simplify(expr_h.coeff(a))

    B_vec = sp.Matrix([c_self, c_far])
    C_vec = sp.Matrix([c_uc, c_near, c_far2])

    beta_hat = sp.simplify(
        bp * (3 - 2 * th) / ((1 - th) * (2 - th))
        + bm * (2 * th + to) / (th * (th + to))
    )

    display(Markdown(f"### Direction: {direction_name}"))
    display(
        Math(
            f"\\theta = {sp.latex(th)},\\quad \\theta_{{\\text{{other}}}} = {sp.latex(to)}"
        )
    )

    # FD coefficients
    # fd_plus = sp.Matrix([sp.simplify(cp0 * h), sp.simplify(cp1 * h), sp.simplify(cp2 * h)])
    # fd_minus = sp.Matrix([sp.simplify(cm0 * h), sp.simplify(cm1 * h), sp.simplify(cm2 * h)])
    # display(Markdown("**FD coefficients for** $u^+_x$ **(interface, near, far2):**"))
    # display(fd_plus)
    # display(Markdown("**FD coefficients for** $u^-_x$ **(far, center, interface):**"))
    # display(fd_minus)

    display(Markdown("**$\\hat\\beta$:**"))
    display(beta_hat)

    # B and C vectors
    display(Markdown("**$B$ vector** (coeffs of $[u^-_D, u^-_{far}]$):"))
    display(sp.Eq(sp.Symbol("B"), B_vec, evaluate=False))
    display(Markdown("**$C$ vector** (coeffs of $[u_c, u_{near}, u_{far2}]$):"))
    display(sp.Eq(sp.Symbol("C"), C_vec, evaluate=False))

    display(Markdown("**$a$ coefficient:**"))
    display(c_a)

    # Verification against the closed-form coefficients used in the solver.
    # Note c_far and c_uc are kept general in theta_other: with th_other = 1
    # they reduce to the case 1/2/3 formulas (uncut opposite side); with
    # th_other < 1 they give the case 4 collinear-pair coupling (u_far is then
    # the opposite interface unknown, an off-diagonal entry of B).
    all_ok = True
    all_ok &= sp.simplify(c_self + beta_hat) == 0
    all_ok &= sp.simplify(c_far - (-bm * th / ((th + to) * to))) == 0
    all_ok &= sp.simplify(c_uc - (bm * (th + to) / (th * to))) == 0
    all_ok &= sp.simplify(c_near - (bp * (2 - th) / (1 - th))) == 0
    all_ok &= sp.simplify(c_far2 - (-bp * (1 - th) / (2 - th))) == 0
    all_ok &= sp.simplify(c_a - (-bp * (3 - 2 * th) / ((1 - th) * (2 - th)))) == 0
    status = "All checks PASSED" if all_ok else "Some checks FAILED"
    display(Markdown(f"**Verification:** {status}"))

    return B_vec, C_vec, c_a, expr_h


# Direction: Right
B_R, C_R, ca_R, expr_R = derive_eq30("Right", th_R, th_L, uRm, uLm, u_m1, u_p1, u_p2, dx)

# Direction: Left
B_L, C_L, ca_L, expr_L = derive_eq30("Left", th_L, th_R, uLm, uRm, u_p1, u_m1, u_m2, dx)

# Direction: Top
B_T, C_T, ca_T, expr_T = derive_eq30("Top", th_T, th_B, uTm, uBm, u_b1, u_t1, u_t2, dy)

# Direction: Bottom
B_B, C_B, ca_B, expr_B = derive_eq30(
    "Bottom", th_B, th_T, uBm, uTm, u_t1, u_b1, u_b2, dy
)


# ===================================================================
# SECTION 2 — Equation (33) for all 4 directions (Case 2)
# ===================================================================
display(Markdown("## Equation (33) — Case 2: two interface points per direction"))


def derive_eq33(
    direction_name,
    th_self,
    th_other,
    th_self2,
    u_self_m,
    u_self2_m,
    u_far_m,
    u_far_grid,
    u_near_grid,
    u_far2_grid,
    u_far3_grid,
    h,
):
    """
    Derive Eq.(33) for a single direction.

    Convention (paper eq. 31): th_self2 is measured from the FAR grid point
    back toward the cell, e.g. for Right the second interface sits at
    x_r = x_{i+2} - th_self2 * dx. In the solver this corresponds to
    compute_theta evaluated at the far cell in the REVERSED direction
    (e.g. compute_theta(Direction::L, i + 2, j)).

    Returns (B_mat, C_mat, a_vec, a2_vec, expr_vec).
      B_mat: 2x2 for [u_self_m, u_self2_m]
      C_mat: 2x5 for [u_far_m, uc, u_near, u_far2, u_far3]
      a_vec, a2_vec: 2x1 jump contributions. Both vectors are dense: the jump
      at EACH collinear interface contributes to BOTH rows (cross terms),
      since u^+ at both interfaces enters each one-sided Omega+ stencil.
    """
    th = th_self
    to = th_other
    thr = th_self2

    # Omega+ side at first interface: {u^+_R, u_{near}, u^+_r} eval at xR=0
    x0_p, x1_p, x2_p = 0, (1 - th) * h, (2 - th - thr) * h
    cp0_atR, cp1_atR, cp2_atR = fd_coeffs_3pt(x0_p, x1_p, x2_p, x0_p)

    # Omega- side at first interface: {u_far_m, uc, u_self_m} eval at xR=0
    x0_m, x1_m, x2_m = -(th + to) * h, -th * h, 0
    cm0_atR, cm1_atR, cm2_atR = fd_coeffs_3pt(x0_m, x1_m, x2_m, x2_m)

    # Omega- side at second interface: {u_self2_m, u_{far2}, u_{far3}} eval at xr=0
    x0_r_m, x1_r_m, x2_r_m = 0, thr * h, (thr + 1) * h
    crm0, crm1, crm2 = fd_coeffs_3pt(x0_r_m, x1_r_m, x2_r_m, x0_r_m)

    # Omega+ side at second interface: {u^+_R, u_{near}, u^+_r} eval at xr
    cxp0, cxp1, cxp2 = fd_coeffs_3pt(x0_p, x1_p, x2_p, x2_p)

    # Explicit u^+ symbols, then substitute u^+_R = u^-_R + a, u^+_r = u^-_r + a_r
    uRp_sym = sp.Symbol("u_R^+")
    uRp2_sym = sp.Symbol("u_{R2}^+")

    # At xR:
    ux_plus_R = cp0_atR * uRp_sym + cp1_atR * u_near_grid + cp2_atR * uRp2_sym
    ux_minus_R = cm0_atR * u_far_m + cm1_atR * uc + cm2_atR * u_self_m
    jump_R = bp * ux_plus_R - bm * ux_minus_R

    # At xr:
    ux_plus_r = cxp0 * uRp_sym + cxp1 * u_near_grid + cxp2 * uRp2_sym
    ux_minus_r = crm0 * u_self2_m + crm1 * u_far2_grid + crm2 * u_far3_grid
    jump_r = bp * ux_plus_r - bm * ux_minus_r

    sub_R = sp.expand(
        (jump_R * h).subs({uRp_sym: u_self_m + a, uRp2_sym: u_self2_m + a2})
    )
    sub_r = sp.expand(
        (jump_r * h).subs({uRp_sym: u_self_m + a, uRp2_sym: u_self2_m + a2})
    )

    B_mat = sp.Matrix(
        [
            [sp.simplify(sub_R.coeff(u_self_m)), sp.simplify(sub_R.coeff(u_self2_m))],
            [sp.simplify(sub_r.coeff(u_self_m)), sp.simplify(sub_r.coeff(u_self2_m))],
        ]
    )
    C_mat = sp.Matrix(
        [
            [
                sp.simplify(sub_R.coeff(u_far_m)),
                sp.simplify(sub_R.coeff(uc)),
                sp.simplify(sub_R.coeff(u_near_grid)),
                sp.simplify(sub_R.coeff(u_far2_grid)),
                sp.simplify(sub_R.coeff(u_far3_grid)),
            ],
            [
                sp.simplify(sub_r.coeff(u_far_m)),
                sp.simplify(sub_r.coeff(uc)),
                sp.simplify(sub_r.coeff(u_near_grid)),
                sp.simplify(sub_r.coeff(u_far2_grid)),
                sp.simplify(sub_r.coeff(u_far3_grid)),
            ],
        ]
    )
    a_vec = sp.Matrix([[sp.simplify(sub_R.coeff(a))], [sp.simplify(sub_r.coeff(a))]])
    a2_vec = sp.Matrix([[sp.simplify(sub_R.coeff(a2))], [sp.simplify(sub_r.coeff(a2))]])

    beta_hat_R = sp.simplify(
        bp * (3 - 2 * th - thr) / ((1 - th) * (2 - th - thr))
        + bm * (2 * th + to) / (th * (th + to))
    )
    beta_hat_r = sp.simplify(
        bm * (2 * thr + 1) / (thr * (thr + 1))
        + bp * (3 - th - 2 * thr) / ((2 - th - thr) * (1 - thr))
    )

    display(Markdown(f"### Direction: {direction_name}"))
    display(
        Math(
            f"\\theta = {sp.latex(th)},\\; "
            f"\\theta_{{\\text{{other}}}} = {sp.latex(to)},\\; "
            f"\\theta_{{\\text{{2nd}}}} = {sp.latex(thr)}"
        )
    )

    # FD coefficients
    fd_plus_atR = sp.Matrix(
        [sp.simplify(cp0_atR * h), sp.simplify(cp1_atR * h), sp.simplify(cp2_atR * h)]
    )
    fd_minus_atR = sp.Matrix(
        [sp.simplify(cm0_atR * h), sp.simplify(cm1_atR * h), sp.simplify(cm2_atR * h)]
    )
    fd_plus_atr = sp.Matrix(
        [sp.simplify(cxp0 * h), sp.simplify(cxp1 * h), sp.simplify(cxp2 * h)]
    )
    fd_minus_atr = sp.Matrix(
        [sp.simplify(crm0 * h), sp.simplify(crm1 * h), sp.simplify(crm2 * h)]
    )

    display(Markdown("**FD coeffs at first interface —** $\\Omega^+$:"))
    display(fd_plus_atR)
    display(Markdown("**FD coeffs at first interface —** $\\Omega^-$:"))
    display(fd_minus_atR)
    display(Markdown("**FD coeffs at second interface —** $\\Omega^+$:"))
    display(fd_plus_atr)
    display(Markdown("**FD coeffs at second interface —** $\\Omega^-$:"))
    display(fd_minus_atr)

    display(Markdown("**$\\hat\\beta_R$:**"))
    display(beta_hat_R)
    display(Markdown("**$\\hat\\beta_r$:**"))
    display(beta_hat_r)

    display(Markdown("**$B$ matrix** $(2\\times 2)$ for $[u^-_D,\\; u^-_{D2}]$:"))
    display(sp.Eq(sp.Symbol("B"), B_mat, evaluate=False))
    display(
        Markdown(
            "**$C$ matrix** $(2\\times 5)$ for $[u^-_{far},\\; u_c,\\; u_{near},\\; u_{far2},\\; u_{far3}]$:"
        )
    )
    display(sp.Eq(sp.Symbol("C"), C_mat, evaluate=False))

    display(Markdown("**$a$ coefficient** (jump at first interface):"))
    display(a_vec)
    display(Markdown("**$a_r$ coefficient** (jump at second interface):"))
    display(a2_vec)

    display(
        Markdown(
            "**Full expression** $[\\text{Eq.(33) at } x_D;\\; \\text{Eq.(33) at } x_{D2}] \\times h$:"
        )
    )
    display(sp.Matrix([sub_R, sub_r]))

    # Verify B[0,0] == -beta_hat_R and B[1,1] == +beta_hat_r.
    # The signs differ because Omega- lies on the cell side of the first
    # interface but on the FAR side of the second one, flipping the one-sided
    # difference orientation (cf. the +beta_hat_r entry in the paper's eq. 33).
    ok_R = sp.simplify(B_mat[0, 0] + beta_hat_R) == 0
    ok_r = sp.simplify(B_mat[1, 1] - beta_hat_r) == 0
    status = "PASSED" if ok_R and ok_r else "FAILED"
    display(
        Markdown(
            f"**Verification:** $B_{{00}} = -\\hat\\beta_R$: {ok_R}, "
            f"$B_{{11}} = +\\hat\\beta_r$: {ok_r}  {status}"
        )
    )

    return B_mat, C_mat, a_vec, a2_vec, sp.Matrix([sub_R, sub_r])


# Direction: Right (Case 2)
B_R2, C_R2, a_R2, a2_R2, expr_R2 = derive_eq33(
    "Right", th_R, th_L, th_R2, uRm, uRm2, uLm, u_m1, u_p1, u_p2, u_p3, dx
)

# Direction: Left (Case 2)
B_L2, C_L2, a_L2, a2_L2, expr_L2 = derive_eq33(
    "Left", th_L, th_R, th_L2, uLm, uLm2, uRm, u_p1, u_m1, u_m2, u_m3, dx
)

# Direction: Top (Case 2)
B_T2, C_T2, a_T2, a2_T2, expr_T2 = derive_eq33(
    "Top", th_T, th_B, th_T2, uTm, uTm2, uBm, u_b1, u_t1, u_t2, u_t3, dy
)

# Direction: Bottom (Case 2)
B_B2, C_B2, a_B2, a2_B2, expr_B2 = derive_eq33(
    "Bottom", th_B, th_T, th_B2, uBm, uBm2, uTm, u_t1, u_b1, u_b2, u_b3, dy
)


# ===================================================================
# Summary
# ===================================================================
display(Markdown("## Summary"))

display(
    Markdown(r"""
**Equation (30) — Case 1** (single interface point per direction):

For each direction $D \in \{R, L, T, B\}$, define $\theta_D$ at the interface
and $\theta_{\text{other}}$ on the far side.

The second-order FD stencil:
- $\Omega^+$ side: $\{u^+_D,\; u_{\text{near}},\; u_{\text{far2}}\}$
- $\Omega^-$ side: $\{u^-_{\text{far}},\; u_c,\; u^-_D\}$

Then:
$$
[\beta u_n]_D \approx
\Big[B_D \cdot (u^-_D,\; u^-_{\text{far}})
      + C_D \cdot (u_c,\; u_{\text{near}},\; u_{\text{far2}})
      - \beta^+_D \frac{3-2\theta_D}{(1-\theta_D)(2-\theta_D)} a_D\Big] \cdot h
$$
where $h = \Delta x$ (for L,R) or $h = \Delta y$ (for T,B).

---

**Equation (33) — Case 2** (two interface points per direction):

Introduce a second interface point $x_{D2}$ with $\theta_{D2}$.
- At first interface $x_D$: $\Omega^+$: $\{u^+_D,\; u_{\text{near}},\; u^+_{D2}\}$, $\Omega^-$: $\{u^-_{\text{far}},\; u_c,\; u^-_D\}$
- At second interface $x_{D2}$: $\Omega^+$: $\{u^+_D,\; u_{\text{near}},\; u^+_{D2}\}$ (diff eval), $\Omega^-$: $\{u^-_{D2},\; u_{\text{far2}},\; u_{\text{far3}}\}$

This gives a $2\!\times\!2$ system:
$$
B_D \begin{bmatrix} u^-_D \\ u^-_{D2} \end{bmatrix}
+ C_D \begin{bmatrix} u^-_{\text{far}} \\ u_c \\ u_{\text{near}} \\ u_{\text{far2}} \\ u_{\text{far3}} \end{bmatrix}
+ a\,\text{-vec}\cdot a_D + a_r\text{-vec}\cdot a_{D2}
= \begin{bmatrix} [\beta u_n]_{x_D} \\ [\beta u_n]_{x_{D2}} \end{bmatrix}
$$
where $B_D \in \mathbb{R}^{2\times 2}$, $C_D \in \mathbb{R}^{2\times 5}$.

---

**Implementation notes** (conventions that are easy to get wrong):

1. $\theta_{D2}$ is measured from the far grid point back toward the cell
   (paper eq. 31): $x_{D2} = x_{i+2} - \theta_{D2}\Delta x$ for direction R.
   In the solver: `compute_theta(Direction::L, i + 2, j)`.
2. Both $a$-vectors are dense (cross terms): the jump at each collinear
   interface contributes to both rows of the system.
3. $\theta_{\text{other}}$ in eq. (30) is 1 only when the opposite side is
   uncut. When it is cut (case 4: three cut directions always contain an
   opposite pair), $u^-_{\text{far}}$ is the opposite interface unknown:
   its coefficient moves from $C$ into an off-diagonal entry of $B$, and
   the grid-point column of $C$ is dropped.
4. Only $u^-_D$ enters the quadratic polynomial $P$ (paper eqs. 26-28) used
   for the geometric jump approximation; $u^-_{D2}$ does not, so no
   $\nabla P$ coupling may be subtracted from its column when assembling M.
""")
)

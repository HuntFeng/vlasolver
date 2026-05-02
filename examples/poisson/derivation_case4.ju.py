# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Case 4 — three corner interfaces around (i,j)
#
# The interface intersects three of the four grid segments emanating from
# the home grid point (i,j). The fourth segment is uncut, so its endpoint
# stays in the same region as (i,j). All three of xR/xT/xL/xB that are
# present sit BETWEEN (i,j) and its immediate neighbour (no extra outer-
# segment interfaces — that's what distinguishes case 4 from case 3).
#
# Sub-cases (eta = sign of phi at (i,j); each gets both eta<0 and eta>0):
# 1. xR, xT, xL    (no B cut)         — figure 18 in the paper
# 2. xR, xT, xB    (no L cut)
# 3. xR, xB, xL    (no T cut)
# 4. xT, xB, xL    (no R cut)
#
# Conventions:
#   xR = x_i + theta_R*dx     xL = x_i - theta_L*dx
#   xT = y_j + theta_T*dy     xB = y_j - theta_B*dy
#
# All polynomial substitutions use OFFSETS from each polynomial's stencil
# center (origin). FD coefficients depend only on dx, dy, theta_R, theta_L,
# theta_T, theta_B — never on absolute x_i / y_j.

# %% jupyter={"source_hidden": true}
import sympy as sp
import sympy.vector as sv
from IPython.display import display
import enum

# %% jupyter={"source_hidden": true}
uT, uB, uL, uR, uc, u_ext = sp.symbols("u_T u_B u_L u_R u_c u_ext")
uT_p, uB_p, uL_p, uR_p, uc_p, u_ext_p = sp.symbols(
    "u_T^+ u_B^+ u_L^+ u_R^+ u_c^+ u_ext^+"
)
uT_m, uB_m, uL_m, uR_m, uc_m, u_ext_m = sp.symbols(
    "u_T^- u_B^- u_L^- u_R^- u_c^- u_ext^-"
)

u = []
for i in range(3):
    u.append(
        [sp.symbols("u_{i%+d\\,j%+d}" % (i, j)) for j in range(3)]
        + [sp.symbols("u_{i%+d\\,j%+d}" % (i, j)) for j in range(-2, 0)]
    )
for i in range(-2, 0):
    u.append(
        [sp.symbols("u_{i%+d\\,j%+d}" % (i, j)) for j in range(3)]
        + [sp.symbols("u_{i%+d\\,j%+d}" % (i, j)) for j in range(-2, 0)]
    )


def flatten(u):
    return [item for sublist in u for item in sublist]


dx, dy = sp.Symbol(r"\Delta x"), sp.Symbol(r"\Delta y")
xL, xR = sp.symbols("x_L x_R")
yT, yB = sp.symbols("y_T y_B")
x_ext, y_ext = sp.symbols("x_ext y_ext")
theta_L, theta_R, theta_T, theta_B = sp.symbols("theta_L theta_R theta_T theta_B")
coord = sv.CoordSys3D("coord")
x, y = coord.x, coord.y

nx, ny = sp.symbols("n_x n_y")
a, a_tau, b = sp.symbols("a, a_{\\tau}, b")
beta_jump, beta_p, beta_m = sp.symbols("[\\beta], beta^+, beta^-")

# Quadratic P(x,y) = A x^2 + B x y + C y^2 + D x + E y + F. Interpolation
# conditions at OFFSETS from the stencil center (origin):
#   P(0, yT)        = uT,  P(0, yB)        = uB,
#   P(xL, 0)        = uL,  P(xR, 0)        = uR,
#   P(0, 0)         = uc,  P(x_ext, y_ext) = u_ext.
A_mat = sp.Matrix(
    [
        [0,        0,            yT**2,    0,     yT,    1],
        [0,        0,            yB**2,    0,     yB,    1],
        [xL**2,    0,            0,        xL,    0,     1],
        [xR**2,    0,            0,        xR,    0,     1],
        [0,        0,            0,        0,     0,     1],
        [x_ext**2, x_ext * y_ext, y_ext**2, x_ext, y_ext, 1],
    ]
)
A_inv = A_mat.inv()
P_coeff = A_inv @ sp.Matrix([uT, uB, uL, uR, uc, u_ext])
A, B, C, D, E, F = P_coeff
P = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F


# %%
class Direction(enum.IntFlag):
    R = 1 << 0
    T = 1 << 1
    L = 1 << 2
    B = 1 << 3


def coeff_case4(eta, points):
    """Build the local 3x3 system M [unknowns]^T = N u + d for case 4.

    `points` is a list of three dicts (one per interface point), each with:
        axis   : 'x' or 'y' — which Cartesian jump condition to enforce
        u_m    : Omega^- ghost symbol at this interface (e.g. uR_m)
        u_p    : Omega^+ ghost symbol at this interface (e.g. uR_p)
        vars_m : substitution dict for the Omega^- polynomial
                 (offsets + value labels + eval point at the interface)
        vars_p : substitution dict for the Omega^+ polynomial

    eta < 0:  home grid point (i,j) is in Omega^- ; unknowns are u_m's.
              [u]=a is used as u_p = u_m + a to eliminate u_p.
    eta > 0:  home grid point is in Omega^+ ; unknowns are u_p's.
              [u]=a is used as u_m = u_p - a to eliminate u_m.

    The implementation is identical to coeff_case3 — case 4 differs from
    case 3 only in the polynomial setup (three interfaces in the home
    Omega^- polynomial, single-interface Omega^+ polynomials per cut).
    """
    grad_P = sv.gradient(P)
    dudx = grad_P.components[coord.i]
    dudy = grad_P.components[coord.j]

    equalities = []
    for pt in points:
        if pt["axis"] == "x":
            if eta < 0:
                geom = (
                    b * nx
                    - beta_jump * ny * (-ny * dudx + nx * dudy)
                    - beta_p * a_tau * ny
                ).subs(pt["vars_m"])
            else:
                geom = (
                    b * nx
                    - beta_jump * ny * (-ny * dudx + nx * dudy)
                    - beta_m * a_tau * ny
                ).subs(pt["vars_p"])
            algebra = beta_p * dudx.subs(pt["vars_p"]) - beta_m * dudx.subs(
                pt["vars_m"]
            )
            eq = (algebra - geom) * dx
        else:  # 'y'
            if eta < 0:
                geom = (
                    b * ny
                    + beta_jump * nx * (-ny * dudx + nx * dudy)
                    + beta_p * a_tau * nx
                ).subs(pt["vars_m"])
            else:
                geom = (
                    b * ny
                    + beta_jump * nx * (-ny * dudx + nx * dudy)
                    + beta_m * a_tau * nx
                ).subs(pt["vars_p"])
            algebra = beta_p * dudy.subs(pt["vars_p"]) - beta_m * dudy.subs(
                pt["vars_m"]
            )
            eq = (algebra - geom) * dy
        equalities.append(eq)

    if eta < 0:
        sub_map = {pt["u_p"]: pt["u_m"] + a for pt in points}
        unknowns = [pt["u_m"] for pt in points]
    else:
        sub_map = {pt["u_m"]: pt["u_p"] - a for pt in points}
        unknowns = [pt["u_p"] for pt in points]

    eq_subs = [eq.subs(sub_map).expand() for eq in equalities]

    M = sp.zeros(3, 3)
    Nu = sp.zeros(3, 1)
    d = sp.zeros(3, 1)

    for i, eq_sub in enumerate(eq_subs):
        for j, sym in enumerate(unknowns):
            c = eq_sub.expand().coeff(sym).collect([beta_p, beta_m, beta_jump])
            num, den = c.as_numer_denom()
            M[i, j] = sp.Add(
                *[(top / den).cancel().factor() for top in num.as_ordered_terms()]
            )

        rest = (eq_sub - sum(M[i, k] * unknowns[k] for k in range(3))).expand()

        u_terms, non_u_terms = [], []
        for term in rest.as_ordered_terms():
            if any(uvar in term.free_symbols for sublist in u for uvar in sublist):
                u_terms.append(term)
            else:
                non_u_terms.append(term)

        d[i] = -sp.Add(
            *[
                term.cancel().factor()
                for term in sp.Add(*non_u_terms)
                .collect([a, b, a_tau])
                .as_ordered_terms()
            ]
        )
        Nu[i] = -sp.Add(
            *[
                term.cancel().factor()
                for term in sp.Add(*u_terms)
                .collect(flatten(u))
                .as_ordered_terms()
            ]
        )

    for i in range(3):
        for j in range(3):
            print(f"M[{i},{j}]")
            display(M[i, j])
    for i in range(3):
        print(f"Nu[{i}]")
        display(Nu[i])
    for i in range(3):
        print(f"d[{i}]")
        display(d[i])

    return M, Nu, d


# %% [markdown]
# Helper: convert an `eta<0` points list into the matching `eta>0` list.
# Same physical interfaces, but now `(i,j) in Omega^+`. The four polynomial
# centers stay put; only their region labels (and therefore the
# `_m` / `_p` ghost symbols) swap.

# %%
_MP_SWAP = {
    uR_m: uR_p, uR_p: uR_m,
    uL_m: uL_p, uL_p: uL_m,
    uT_m: uT_p, uT_p: uT_m,
    uB_m: uB_p, uB_p: uB_m,
}


def mp_swap(d):
    return {k: _MP_SWAP.get(v, v) for k, v in d.items()}


def points_for_eta_pos(points_neg):
    return [
        {
            "axis": pt["axis"],
            "u_m": pt["u_m"],
            "u_p": pt["u_p"],
            "vars_m": mp_swap(pt["vars_p"]),
            "vars_p": mp_swap(pt["vars_m"]),
        }
        for pt in points_neg
    ]


# %% [markdown]
# ## Sub-case 1: xR, xT, xL  (no B cut)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i,j-1) in Omega^- (uncut B side),
#   (i+1,j), (i-1,j), (i,j+1) in Omega^+.
# Polynomials:
#   m@(i,j)   carries xR (xR slot) + xT (yT slot) + xL (xL slot);
#             uB = u_{i,j-1} (in Omega^-), u_ext = u_{i-1,j-1} (in Omega^-).
#   p@(i+1,j) for xR  (xR sits at xL slot, offset -(1-theta_R)*dx)
#   p@(i-1,j) for xL  (xL sits at xR slot, offset +(1-theta_L)*dx)
#   p@(i,j+1) for xT  (xT sits at yB slot, offset -(1-theta_T)*dy)

# %%
s1_pm_ij = {
    xL: -theta_L * dx, xR: theta_R * dx, yT: theta_T * dy, yB: -dy,
    x_ext: -dx, y_ext: -dy,
    uL: uL_m, uR: uR_m, uT: uT_m, uB: u[0][-1],
    uc: u[0][0], u_ext: u[-1][-1],
}
s1_pp_i1j = {
    xL: -(1 - theta_R) * dx, xR: dx, yT: dy, yB: -dy,
    x_ext: dx, y_ext: dy,
    uL: uR_p, uR: u[2][0], uT: u[1][1], uB: u[1][-1],
    uc: u[1][0], u_ext: u[2][1],
}
s1_pp_im1j = {
    xL: -dx, xR: (1 - theta_L) * dx, yT: dy, yB: -dy,
    x_ext: -dx, y_ext: dy,
    uL: u[-2][0], uR: uL_p, uT: u[-1][1], uB: u[-1][-1],
    uc: u[-1][0], u_ext: u[-2][1],
}
s1_pp_ij1 = {
    xL: -dx, xR: dx, yT: dy, yB: -(1 - theta_T) * dy,
    x_ext: dx, y_ext: dy,
    uL: u[-1][1], uR: u[1][1], uT: u[0][2], uB: uT_p,
    uc: u[0][1], u_ext: u[1][2],
}

s1_pts_neg = [
    {"axis": "x", "u_m": uR_m, "u_p": uR_p,
     "vars_m": {x: theta_R * dx, y: 0, **s1_pm_ij},
     "vars_p": {x: -(1 - theta_R) * dx, y: 0, **s1_pp_i1j}},
    {"axis": "y", "u_m": uT_m, "u_p": uT_p,
     "vars_m": {x: 0, y: theta_T * dy, **s1_pm_ij},
     "vars_p": {x: 0, y: -(1 - theta_T) * dy, **s1_pp_ij1}},
    {"axis": "x", "u_m": uL_m, "u_p": uL_p,
     "vars_m": {x: -theta_L * dx, y: 0, **s1_pm_ij},
     "vars_p": {x: (1 - theta_L) * dx, y: 0, **s1_pp_im1j}},
]

# %%
print("=== Sub-case 1, eta < 0 ===")
M, Nu, d = coeff_case4(-1, s1_pts_neg)

# %%
print("=== Sub-case 1, eta > 0 ===")
M, Nu, d = coeff_case4(+1, points_for_eta_pos(s1_pts_neg))


# %% [markdown]
# ## Sub-case 2: xR, xT, xB  (no L cut)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i-1,j) in Omega^- (uncut L side),
#   (i+1,j), (i,j+1), (i,j-1) in Omega^+.
# Polynomials:
#   m@(i,j)   carries xR + xT + xB; uL = u_{i-1,j} (in Omega^-),
#             u_ext = u_{i-1,j+1} (in Omega^-, on the uncut diagonal).
#   p@(i+1,j) for xR
#   p@(i,j+1) for xT
#   p@(i,j-1) for xB  (xB sits at yT slot, offset +(1-theta_B)*dy)

# %%
s2_pm_ij = {
    xL: -dx, xR: theta_R * dx, yT: theta_T * dy, yB: -theta_B * dy,
    x_ext: -dx, y_ext: dy,
    uL: u[-1][0], uR: uR_m, uT: uT_m, uB: uB_m,
    uc: u[0][0], u_ext: u[-1][1],
}
s2_pp_i1j = {
    xL: -(1 - theta_R) * dx, xR: dx, yT: dy, yB: -dy,
    x_ext: dx, y_ext: dy,
    uL: uR_p, uR: u[2][0], uT: u[1][1], uB: u[1][-1],
    uc: u[1][0], u_ext: u[2][1],
}
s2_pp_ij1 = {
    xL: -dx, xR: dx, yT: dy, yB: -(1 - theta_T) * dy,
    x_ext: dx, y_ext: dy,
    uL: u[-1][1], uR: u[1][1], uT: u[0][2], uB: uT_p,
    uc: u[0][1], u_ext: u[1][2],
}
s2_pp_ijm1 = {
    xL: -dx, xR: dx, yT: (1 - theta_B) * dy, yB: -dy,
    x_ext: dx, y_ext: -dy,
    uL: u[-1][-1], uR: u[1][-1], uT: uB_p, uB: u[0][-2],
    uc: u[0][-1], u_ext: u[1][-2],
}

s2_pts_neg = [
    {"axis": "x", "u_m": uR_m, "u_p": uR_p,
     "vars_m": {x: theta_R * dx, y: 0, **s2_pm_ij},
     "vars_p": {x: -(1 - theta_R) * dx, y: 0, **s2_pp_i1j}},
    {"axis": "y", "u_m": uT_m, "u_p": uT_p,
     "vars_m": {x: 0, y: theta_T * dy, **s2_pm_ij},
     "vars_p": {x: 0, y: -(1 - theta_T) * dy, **s2_pp_ij1}},
    {"axis": "y", "u_m": uB_m, "u_p": uB_p,
     "vars_m": {x: 0, y: -theta_B * dy, **s2_pm_ij},
     "vars_p": {x: 0, y: (1 - theta_B) * dy, **s2_pp_ijm1}},
]

# %%
print("=== Sub-case 2, eta < 0 ===")
M, Nu, d = coeff_case4(-1, s2_pts_neg)

# %%
print("=== Sub-case 2, eta > 0 ===")
M, Nu, d = coeff_case4(+1, points_for_eta_pos(s2_pts_neg))


# %% [markdown]
# ## Sub-case 3: xR, xB, xL  (no T cut)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i,j+1) in Omega^- (uncut T side),
#   (i+1,j), (i-1,j), (i,j-1) in Omega^+.
# Polynomials:
#   m@(i,j)   carries xR + xB + xL; uT = u_{i,j+1} (in Omega^-),
#             u_ext = u_{i-1,j+1} (in Omega^-).
#   p@(i+1,j) for xR
#   p@(i-1,j) for xL
#   p@(i,j-1) for xB

# %%
s3_pm_ij = {
    xL: -theta_L * dx, xR: theta_R * dx, yT: dy, yB: -theta_B * dy,
    x_ext: -dx, y_ext: dy,
    uL: uL_m, uR: uR_m, uT: u[0][1], uB: uB_m,
    uc: u[0][0], u_ext: u[-1][1],
}
s3_pp_i1j = {
    xL: -(1 - theta_R) * dx, xR: dx, yT: dy, yB: -dy,
    x_ext: dx, y_ext: -dy,
    uL: uR_p, uR: u[2][0], uT: u[1][1], uB: u[1][-1],
    uc: u[1][0], u_ext: u[2][-1],
}
s3_pp_im1j = {
    xL: -dx, xR: (1 - theta_L) * dx, yT: dy, yB: -dy,
    x_ext: -dx, y_ext: -dy,
    uL: u[-2][0], uR: uL_p, uT: u[-1][1], uB: u[-1][-1],
    uc: u[-1][0], u_ext: u[-2][-1],
}
s3_pp_ijm1 = {
    xL: -dx, xR: dx, yT: (1 - theta_B) * dy, yB: -dy,
    x_ext: -dx, y_ext: -dy,
    uL: u[-1][-1], uR: u[1][-1], uT: uB_p, uB: u[0][-2],
    uc: u[0][-1], u_ext: u[-1][-2],
}

s3_pts_neg = [
    {"axis": "x", "u_m": uR_m, "u_p": uR_p,
     "vars_m": {x: theta_R * dx, y: 0, **s3_pm_ij},
     "vars_p": {x: -(1 - theta_R) * dx, y: 0, **s3_pp_i1j}},
    {"axis": "y", "u_m": uB_m, "u_p": uB_p,
     "vars_m": {x: 0, y: -theta_B * dy, **s3_pm_ij},
     "vars_p": {x: 0, y: (1 - theta_B) * dy, **s3_pp_ijm1}},
    {"axis": "x", "u_m": uL_m, "u_p": uL_p,
     "vars_m": {x: -theta_L * dx, y: 0, **s3_pm_ij},
     "vars_p": {x: (1 - theta_L) * dx, y: 0, **s3_pp_im1j}},
]

# %%
print("=== Sub-case 3, eta < 0 ===")
M, Nu, d = coeff_case4(-1, s3_pts_neg)

# %%
print("=== Sub-case 3, eta > 0 ===")
M, Nu, d = coeff_case4(+1, points_for_eta_pos(s3_pts_neg))


# %% [markdown]
# ## Sub-case 4: xT, xB, xL  (no R cut)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i+1,j) in Omega^- (uncut R side),
#   (i-1,j), (i,j+1), (i,j-1) in Omega^+.
# Polynomials:
#   m@(i,j)   carries xL + xT + xB; uR = u_{i+1,j} (in Omega^-),
#             u_ext = u_{i+1,j-1} (in Omega^-).
#   p@(i-1,j) for xL
#   p@(i,j+1) for xT
#   p@(i,j-1) for xB

# %%
s4_pm_ij = {
    xL: -theta_L * dx, xR: dx, yT: theta_T * dy, yB: -theta_B * dy,
    x_ext: dx, y_ext: -dy,
    uL: uL_m, uR: u[1][0], uT: uT_m, uB: uB_m,
    uc: u[0][0], u_ext: u[1][-1],
}
s4_pp_im1j = {
    xL: -dx, xR: (1 - theta_L) * dx, yT: dy, yB: -dy,
    x_ext: -dx, y_ext: dy,
    uL: u[-2][0], uR: uL_p, uT: u[-1][1], uB: u[-1][-1],
    uc: u[-1][0], u_ext: u[-2][1],
}
s4_pp_ij1 = {
    xL: -dx, xR: dx, yT: dy, yB: -(1 - theta_T) * dy,
    x_ext: -dx, y_ext: dy,
    uL: u[-1][1], uR: u[1][1], uT: u[0][2], uB: uT_p,
    uc: u[0][1], u_ext: u[-1][2],
}
s4_pp_ijm1 = {
    xL: -dx, xR: dx, yT: (1 - theta_B) * dy, yB: -dy,
    x_ext: -dx, y_ext: -dy,
    uL: u[-1][-1], uR: u[1][-1], uT: uB_p, uB: u[0][-2],
    uc: u[0][-1], u_ext: u[-1][-2],
}

s4_pts_neg = [
    {"axis": "y", "u_m": uT_m, "u_p": uT_p,
     "vars_m": {x: 0, y: theta_T * dy, **s4_pm_ij},
     "vars_p": {x: 0, y: -(1 - theta_T) * dy, **s4_pp_ij1}},
    {"axis": "y", "u_m": uB_m, "u_p": uB_p,
     "vars_m": {x: 0, y: -theta_B * dy, **s4_pm_ij},
     "vars_p": {x: 0, y: (1 - theta_B) * dy, **s4_pp_ijm1}},
    {"axis": "x", "u_m": uL_m, "u_p": uL_p,
     "vars_m": {x: -theta_L * dx, y: 0, **s4_pm_ij},
     "vars_p": {x: (1 - theta_L) * dx, y: 0, **s4_pp_im1j}},
]

# %%
print("=== Sub-case 4, eta < 0 ===")
M, Nu, d = coeff_case4(-1, s4_pts_neg)

# %%
print("=== Sub-case 4, eta > 0 ===")
M, Nu, d = coeff_case4(+1, points_for_eta_pos(s4_pts_neg))

# %%

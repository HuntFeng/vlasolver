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
# ### Case 3 — second-order GFM stencils with three interface points
#
# Sub-cases (eta = sign of phi at the home grid point (i,j)):
# 1. xR, xT, extra xr   (top-right + extra right)
# 2. xR, xT, extra xt   (top-right + extra top)
# 3. xT, xL, extra xt   (top-left  + extra top)
# 4. xT, xL, extra xl   (top-left  + extra left)
# 5. xL, xB, extra xl   (bot-left  + extra left)
# 6. xL, xB, extra xb   (bot-left  + extra bottom)
# 7. xB, xR, extra xb   (bot-right + extra bottom)
# 8. xB, xR, extra xr   (bot-right + extra right)
#
# Conventions (paper):
#   xR = x_i + theta_R * dx    xL = x_i - theta_L * dx
#   xT = y_j + theta_T * dy    xB = y_j - theta_B * dy
#   xr = x_{i+2} - theta_r*dx  xl = x_{i-2} + theta_l*dx
#   xt = y_{j+2} - theta_t*dy  xb = y_{j-2} + theta_b*dy
#
# All polynomial substitutions use OFFSETS from each polynomial's stencil
# center (origin), so the resulting FD coefficients depend only on
# dx, dy, theta_R, theta_T, theta_r, ... — never on absolute x_i / y_j.

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

# Extra interface points (lowercase to distinguish from xR/xT/xL/xB).
ur, ut, ul, ub = sp.symbols("u_r u_t u_l u_b")
ur_p, ut_p, ul_p, ub_p = sp.symbols("u_r^+ u_t^+ u_l^+ u_b^+")
ur_m, ut_m, ul_m, ub_m = sp.symbols("u_r^- u_t^- u_l^- u_b^-")

u = []
for i in range(4):
    u.append(
        [sp.symbols("u_{i%+d\\,j%+d}" % (i, j)) for j in range(4)]
        + [sp.symbols("u_{i%+d\\,j%+d}" % (i, j)) for j in range(-3, 0)]
    )
for i in range(-3, 0):
    u.append(
        [sp.symbols("u_{i%+d\\,j%+d}" % (i, j)) for j in range(4)]
        + [sp.symbols("u_{i%+d\\,j%+d}" % (i, j)) for j in range(-3, 0)]
    )


def flatten(u):
    return [item for sublist in u for item in sublist]


dx, dy = sp.Symbol(r"\Delta x"), sp.Symbol(r"\Delta y")
xL, xR = sp.symbols("x_L x_R")
yT, yB = sp.symbols("y_T y_B")
x_ext, y_ext = sp.symbols("x_ext y_ext")
theta_L, theta_R, theta_T, theta_B = sp.symbols("theta_L theta_R theta_T theta_B")
theta_r, theta_t, theta_l, theta_b = sp.symbols("theta_r theta_t theta_l theta_b")
coord = sv.CoordSys3D("coord")
x, y = coord.x, coord.y

nx, ny = sp.symbols("n_x n_y")
a, a_tau, b = sp.symbols("a, a_{\\tau}, b")
beta_jump, beta_p, beta_m = sp.symbols("[\\beta], beta^+, beta^-")

# Quadratic P(x,y) = A x^2 + B x y + C y^2 + D x + E y + F.
# Conditions interpolate at OFFSETS from the stencil center (origin):
#   P(0, yT)        = uT,  P(0, yB)        = uB,
#   P(xL, 0)        = uL,  P(xR, 0)        = uR,
#   P(0, 0)         = uc,  P(x_ext, y_ext) = u_ext.
A_mat = sp.Matrix(
    [
        [0, 0, yT**2, 0, yT, 1],
        [0, 0, yB**2, 0, yB, 1],
        [xL**2, 0, 0, xL, 0, 1],
        [xR**2, 0, 0, xR, 0, 1],
        [0, 0, 0, 0, 0, 1],
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


def coeff_case3(eta, points):
    """Build the local 3x3 system M [unknowns]^T = N u + d for case 3.

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
                for term in sp.Add(*u_terms).collect(flatten(u)).as_ordered_terms()
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
    uR_m: uR_p,
    uR_p: uR_m,
    uL_m: uL_p,
    uL_p: uL_m,
    uT_m: uT_p,
    uT_p: uT_m,
    uB_m: uB_p,
    uB_p: uB_m,
    ur_m: ur_p,
    ur_p: ur_m,
    ut_m: ut_p,
    ut_p: ut_m,
    ul_m: ul_p,
    ul_p: ul_m,
    ub_m: ub_p,
    ub_p: ub_m,
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
# ## Sub-case 1: xR, xT, extra xr  (top-right + extra right)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i+1,j) in Omega^+, (i+2,j) in Omega^-, (i,j+1) in Omega^+.
# Polynomials:
#   m@(i,j)   carries xR (xR slot) + xT (yT slot)
#   m@(i+2,j) carries xr (xL slot)
#   p@(i+1,j) carries xR (xL slot) + xr (xR slot)   — both interfaces on x-axis
#   p@(i,j+1) carries xT (yB slot)

# %%
s1_pm_ij = {
    xL: -dx,
    xR: theta_R * dx,
    yT: theta_T * dy,
    yB: -dy,
    x_ext: -dx,
    y_ext: -dy,
    uL: u[-1][0],
    uR: uR_m,
    uT: uT_m,
    uB: u[0][-1],
    uc: u[0][0],
    u_ext: u[-1][-1],
}
s1_pm_i2j = {
    xL: -theta_r * dx,
    xR: dx,
    yT: dy,
    yB: -dy,
    x_ext: dx,
    y_ext: -dy,
    uL: ur_m,
    uR: u[3][0],
    uT: u[2][1],
    uB: u[2][-1],
    uc: u[2][0],
    u_ext: u[3][-1],
}
s1_pp_i1j = {
    xL: -(1 - theta_R) * dx,
    xR: (1 - theta_r) * dx,
    yT: dy,
    yB: -dy,
    x_ext: dx,
    y_ext: dy,
    uL: uR_p,
    uR: ur_p,
    uT: u[1][1],
    uB: u[1][-1],
    uc: u[1][0],
    u_ext: u[2][1],
}
s1_pp_ij1 = {
    xL: -dx,
    xR: dx,
    yT: dy,
    yB: -(1 - theta_T) * dy,
    x_ext: dx,
    y_ext: dy,
    uL: u[-1][1],
    uR: u[1][1],
    uT: u[0][2],
    uB: uT_p,
    uc: u[0][1],
    u_ext: u[1][2],
}

s1_pts_neg = [
    {
        "axis": "x",
        "u_m": uR_m,
        "u_p": uR_p,
        "vars_m": {x: theta_R * dx, y: 0, **s1_pm_ij},
        "vars_p": {x: -(1 - theta_R) * dx, y: 0, **s1_pp_i1j},
    },
    {
        "axis": "x",
        "u_m": ur_m,
        "u_p": ur_p,
        "vars_m": {x: -theta_r * dx, y: 0, **s1_pm_i2j},
        "vars_p": {x: (1 - theta_r) * dx, y: 0, **s1_pp_i1j},
    },
    {
        "axis": "y",
        "u_m": uT_m,
        "u_p": uT_p,
        "vars_m": {x: 0, y: theta_T * dy, **s1_pm_ij},
        "vars_p": {x: 0, y: -(1 - theta_T) * dy, **s1_pp_ij1},
    },
]

# %%
print("=== Sub-case 1, eta < 0 ===")
M, Nu, d = coeff_case3(-1, s1_pts_neg)

# %%
print("=== Sub-case 1, eta > 0 ===")
M, Nu, d = coeff_case3(+1, points_for_eta_pos(s1_pts_neg))


# %% [markdown]
# ## Sub-case 2: xR, xT, extra xt  (top-right + extra top)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i+1,j) in Omega^+, (i,j+1) in Omega^+, (i,j+2) in Omega^-.
# Polynomials:
#   m@(i,j)   carries xR + xT     (same as sub-case 1)
#   m@(i,j+2) carries xt (yB slot)
#   p@(i+1,j) carries xR (xL slot)
#   p@(i,j+1) carries xT (yB slot) + xt (yT slot)   — both on y-axis

# %%
s2_pm_ij = s1_pm_ij

s2_pm_ij2 = {
    xL: -dx,
    xR: dx,
    yT: dy,
    yB: -theta_t * dy,
    x_ext: -dx,
    y_ext: dy,
    uL: u[-1][2],
    uR: u[1][2],
    uT: u[0][3],
    uB: ut_m,
    uc: u[0][2],
    u_ext: u[-1][3],
}
s2_pp_i1j = {
    xL: -(1 - theta_R) * dx,
    xR: dx,
    yT: dy,
    yB: -dy,
    x_ext: dx,
    y_ext: dy,
    uL: uR_p,
    uR: u[2][0],
    uT: u[1][1],
    uB: u[1][-1],
    uc: u[1][0],
    u_ext: u[2][1],
}
s2_pp_ij1 = {
    xL: -dx,
    xR: dx,
    yT: (1 - theta_t) * dy,
    yB: -(1 - theta_T) * dy,
    x_ext: dx,
    y_ext: dy,
    uL: u[-1][1],
    uR: u[1][1],
    uT: ut_p,
    uB: uT_p,
    uc: u[0][1],
    u_ext: u[1][2],
}

s2_pts_neg = [
    {
        "axis": "x",
        "u_m": uR_m,
        "u_p": uR_p,
        "vars_m": {x: theta_R * dx, y: 0, **s2_pm_ij},
        "vars_p": {x: -(1 - theta_R) * dx, y: 0, **s2_pp_i1j},
    },
    {
        "axis": "y",
        "u_m": uT_m,
        "u_p": uT_p,
        "vars_m": {x: 0, y: theta_T * dy, **s2_pm_ij},
        "vars_p": {x: 0, y: -(1 - theta_T) * dy, **s2_pp_ij1},
    },
    {
        "axis": "y",
        "u_m": ut_m,
        "u_p": ut_p,
        "vars_m": {x: 0, y: -theta_t * dy, **s2_pm_ij2},
        "vars_p": {x: 0, y: (1 - theta_t) * dy, **s2_pp_ij1},
    },
]

# %%
print("=== Sub-case 2, eta < 0 ===")
M, Nu, d = coeff_case3(-1, s2_pts_neg)

# %%
print("=== Sub-case 2, eta > 0 ===")
M, Nu, d = coeff_case3(+1, points_for_eta_pos(s2_pts_neg))


# %% [markdown]
# ## Sub-case 3: xT, xL, extra xt  (top-left + extra top)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i-1,j) in Omega^+, (i,j+1) in Omega^+, (i,j+2) in Omega^-.
# Polynomials:
#   m@(i,j)   carries xL (xL slot) + xT (yT slot)
#   m@(i,j+2) carries xt   (same shape as sub-case 2)
#   p@(i-1,j) carries xL (xR slot)
#   p@(i,j+1) carries xT + xt   (same as sub-case 2)

# %%
s3_pm_ij = {
    xL: -theta_L * dx,
    xR: dx,
    yT: theta_T * dy,
    yB: -dy,
    x_ext: dx,
    y_ext: -dy,
    uL: uL_m,
    uR: u[1][0],
    uT: uT_m,
    uB: u[0][-1],
    uc: u[0][0],
    u_ext: u[1][-1],
}
s3_pm_ij2 = s2_pm_ij2
s3_pp_im1j = {
    xL: -dx,
    xR: (1 - theta_L) * dx,
    yT: dy,
    yB: -dy,
    x_ext: -dx,
    y_ext: dy,
    uL: u[-2][0],
    uR: uL_p,
    uT: u[-1][1],
    uB: u[-1][-1],
    uc: u[-1][0],
    u_ext: u[-2][1],
}
s3_pp_ij1 = s2_pp_ij1

s3_pts_neg = [
    {
        "axis": "y",
        "u_m": uT_m,
        "u_p": uT_p,
        "vars_m": {x: 0, y: theta_T * dy, **s3_pm_ij},
        "vars_p": {x: 0, y: -(1 - theta_T) * dy, **s3_pp_ij1},
    },
    {
        "axis": "x",
        "u_m": uL_m,
        "u_p": uL_p,
        "vars_m": {x: -theta_L * dx, y: 0, **s3_pm_ij},
        "vars_p": {x: (1 - theta_L) * dx, y: 0, **s3_pp_im1j},
    },
    {
        "axis": "y",
        "u_m": ut_m,
        "u_p": ut_p,
        "vars_m": {x: 0, y: -theta_t * dy, **s3_pm_ij2},
        "vars_p": {x: 0, y: (1 - theta_t) * dy, **s3_pp_ij1},
    },
]

# %%
print("=== Sub-case 3, eta < 0 ===")
M, Nu, d = coeff_case3(-1, s3_pts_neg)

# %%
print("=== Sub-case 3, eta > 0 ===")
M, Nu, d = coeff_case3(+1, points_for_eta_pos(s3_pts_neg))


# %% [markdown]
# ## Sub-case 4: xT, xL, extra xl  (top-left + extra left)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i-1,j) in Omega^+, (i,j+1) in Omega^+, (i-2,j) in Omega^-.
# Polynomials:
#   m@(i,j)   carries xL + xT (same as sub-case 3)
#   m@(i-2,j) carries xl (xR slot, since xl = x_{i-2}+theta_l*dx)
#   p@(i-1,j) carries xL (xR slot) + xl (xL slot)
#   p@(i,j+1) carries xT only

# %%
s4_pm_ij = s3_pm_ij
s4_pm_im2j = {
    xL: -dx,
    xR: theta_l * dx,
    yT: dy,
    yB: -dy,
    x_ext: -dx,
    y_ext: -dy,
    uL: u[-3][0],
    uR: ul_m,
    uT: u[-2][1],
    uB: u[-2][-1],
    uc: u[-2][0],
    u_ext: u[-3][-1],
}
s4_pp_im1j = {
    xL: -(1 - theta_l) * dx,
    xR: (1 - theta_L) * dx,
    yT: dy,
    yB: -dy,
    x_ext: -dx,
    y_ext: dy,
    uL: ul_p,
    uR: uL_p,
    uT: u[-1][1],
    uB: u[-1][-1],
    uc: u[-1][0],
    u_ext: u[-2][1],
}
s4_pp_ij1 = {
    xL: -dx,
    xR: dx,
    yT: dy,
    yB: -(1 - theta_T) * dy,
    x_ext: -dx,
    y_ext: dy,
    uL: u[-1][1],
    uR: u[1][1],
    uT: u[0][2],
    uB: uT_p,
    uc: u[0][1],
    u_ext: u[-1][2],
}

s4_pts_neg = [
    {
        "axis": "y",
        "u_m": uT_m,
        "u_p": uT_p,
        "vars_m": {x: 0, y: theta_T * dy, **s4_pm_ij},
        "vars_p": {x: 0, y: -(1 - theta_T) * dy, **s4_pp_ij1},
    },
    {
        "axis": "x",
        "u_m": uL_m,
        "u_p": uL_p,
        "vars_m": {x: -theta_L * dx, y: 0, **s4_pm_ij},
        "vars_p": {x: (1 - theta_L) * dx, y: 0, **s4_pp_im1j},
    },
    {
        "axis": "x",
        "u_m": ul_m,
        "u_p": ul_p,
        "vars_m": {x: theta_l * dx, y: 0, **s4_pm_im2j},
        "vars_p": {x: -(1 - theta_l) * dx, y: 0, **s4_pp_im1j},
    },
]

# %%
print("=== Sub-case 4, eta < 0 ===")
M, Nu, d = coeff_case3(-1, s4_pts_neg)

# %%
print("=== Sub-case 4, eta > 0 ===")
M, Nu, d = coeff_case3(+1, points_for_eta_pos(s4_pts_neg))


# %% [markdown]
# ## Sub-case 5: xL, xB, extra xl  (bottom-left + extra left)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i-1,j) in Omega^+, (i,j-1) in Omega^+, (i-2,j) in Omega^-.

# %%
s5_pm_ij = {
    xL: -theta_L * dx,
    xR: dx,
    yT: dy,
    yB: -theta_B * dy,
    x_ext: dx,
    y_ext: dy,
    uL: uL_m,
    uR: u[1][0],
    uT: u[0][1],
    uB: uB_m,
    uc: u[0][0],
    u_ext: u[1][1],
}
s5_pm_im2j = s4_pm_im2j
s5_pp_im1j = {
    xL: -(1 - theta_l) * dx,
    xR: (1 - theta_L) * dx,
    yT: dy,
    yB: -dy,
    x_ext: -dx,
    y_ext: -dy,
    uL: ul_p,
    uR: uL_p,
    uT: u[-1][1],
    uB: u[-1][-1],
    uc: u[-1][0],
    u_ext: u[-2][-1],
}
s5_pp_ijm1 = {
    xL: -dx,
    xR: dx,
    yT: (1 - theta_B) * dy,
    yB: -dy,
    x_ext: -dx,
    y_ext: -dy,
    uL: u[-1][-1],
    uR: u[1][-1],
    uT: uB_p,
    uB: u[0][-2],
    uc: u[0][-1],
    u_ext: u[-1][-2],
}

s5_pts_neg = [
    {
        "axis": "x",
        "u_m": uL_m,
        "u_p": uL_p,
        "vars_m": {x: -theta_L * dx, y: 0, **s5_pm_ij},
        "vars_p": {x: (1 - theta_L) * dx, y: 0, **s5_pp_im1j},
    },
    {
        "axis": "y",
        "u_m": uB_m,
        "u_p": uB_p,
        "vars_m": {x: 0, y: -theta_B * dy, **s5_pm_ij},
        "vars_p": {x: 0, y: (1 - theta_B) * dy, **s5_pp_ijm1},
    },
    {
        "axis": "x",
        "u_m": ul_m,
        "u_p": ul_p,
        "vars_m": {x: theta_l * dx, y: 0, **s5_pm_im2j},
        "vars_p": {x: -(1 - theta_l) * dx, y: 0, **s5_pp_im1j},
    },
]

# %%
print("=== Sub-case 5, eta < 0 ===")
M, Nu, d = coeff_case3(-1, s5_pts_neg)

# %%
print("=== Sub-case 5, eta > 0 ===")
M, Nu, d = coeff_case3(+1, points_for_eta_pos(s5_pts_neg))


# %% [markdown]
# ## Sub-case 6: xL, xB, extra xb  (bottom-left + extra bottom)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i-1,j) in Omega^+, (i,j-1) in Omega^+, (i,j-2) in Omega^-.

# %%
s6_pm_ij = s5_pm_ij
s6_pm_ijm2 = {
    xL: -dx,
    xR: dx,
    yT: theta_b * dy,
    yB: -dy,
    x_ext: -dx,
    y_ext: -dy,
    uL: u[-1][-2],
    uR: u[1][-2],
    uT: ub_m,
    uB: u[0][-3],
    uc: u[0][-2],
    u_ext: u[-1][-3],
}
s6_pp_im1j = {
    xL: -dx,
    xR: (1 - theta_L) * dx,
    yT: dy,
    yB: -dy,
    x_ext: -dx,
    y_ext: -dy,
    uL: u[-2][0],
    uR: uL_p,
    uT: u[-1][1],
    uB: u[-1][-1],
    uc: u[-1][0],
    u_ext: u[-2][-1],
}
s6_pp_ijm1 = {
    xL: -dx,
    xR: dx,
    yT: (1 - theta_B) * dy,
    yB: -(1 - theta_b) * dy,
    x_ext: -dx,
    y_ext: -dy,
    uL: u[-1][-1],
    uR: u[1][-1],
    uT: uB_p,
    uB: ub_p,
    uc: u[0][-1],
    u_ext: u[-1][-2],
}

s6_pts_neg = [
    {
        "axis": "x",
        "u_m": uL_m,
        "u_p": uL_p,
        "vars_m": {x: -theta_L * dx, y: 0, **s6_pm_ij},
        "vars_p": {x: (1 - theta_L) * dx, y: 0, **s6_pp_im1j},
    },
    {
        "axis": "y",
        "u_m": uB_m,
        "u_p": uB_p,
        "vars_m": {x: 0, y: -theta_B * dy, **s6_pm_ij},
        "vars_p": {x: 0, y: (1 - theta_B) * dy, **s6_pp_ijm1},
    },
    {
        "axis": "y",
        "u_m": ub_m,
        "u_p": ub_p,
        "vars_m": {x: 0, y: theta_b * dy, **s6_pm_ijm2},
        "vars_p": {x: 0, y: -(1 - theta_b) * dy, **s6_pp_ijm1},
    },
]

# %%
print("=== Sub-case 6, eta < 0 ===")
M, Nu, d = coeff_case3(-1, s6_pts_neg)

# %%
print("=== Sub-case 6, eta > 0 ===")
M, Nu, d = coeff_case3(+1, points_for_eta_pos(s6_pts_neg))


# %% [markdown]
# ## Sub-case 7: xB, xR, extra xb  (bottom-right + extra bottom)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i+1,j) in Omega^+, (i,j-1) in Omega^+, (i,j-2) in Omega^-.

# %%
s7_pm_ij = {
    xL: -dx,
    xR: theta_R * dx,
    yT: dy,
    yB: -theta_B * dy,
    x_ext: -dx,
    y_ext: dy,
    uL: u[-1][0],
    uR: uR_m,
    uT: u[0][1],
    uB: uB_m,
    uc: u[0][0],
    u_ext: u[-1][1],
}
s7_pm_ijm2 = s6_pm_ijm2
s7_pp_i1j = {
    xL: -(1 - theta_R) * dx,
    xR: dx,
    yT: dy,
    yB: -dy,
    x_ext: dx,
    y_ext: -dy,
    uL: uR_p,
    uR: u[2][0],
    uT: u[1][1],
    uB: u[1][-1],
    uc: u[1][0],
    u_ext: u[2][-1],
}
s7_pp_ijm1 = s6_pp_ijm1

s7_pts_neg = [
    {
        "axis": "y",
        "u_m": uB_m,
        "u_p": uB_p,
        "vars_m": {x: 0, y: -theta_B * dy, **s7_pm_ij},
        "vars_p": {x: 0, y: (1 - theta_B) * dy, **s7_pp_ijm1},
    },
    {
        "axis": "x",
        "u_m": uR_m,
        "u_p": uR_p,
        "vars_m": {x: theta_R * dx, y: 0, **s7_pm_ij},
        "vars_p": {x: -(1 - theta_R) * dx, y: 0, **s7_pp_i1j},
    },
    {
        "axis": "y",
        "u_m": ub_m,
        "u_p": ub_p,
        "vars_m": {x: 0, y: theta_b * dy, **s7_pm_ijm2},
        "vars_p": {x: 0, y: -(1 - theta_b) * dy, **s7_pp_ijm1},
    },
]

# %%
print("=== Sub-case 7, eta < 0 ===")
M, Nu, d = coeff_case3(-1, s7_pts_neg)

# %%
print("=== Sub-case 7, eta > 0 ===")
M, Nu, d = coeff_case3(+1, points_for_eta_pos(s7_pts_neg))


# %% [markdown]
# ## Sub-case 8: xB, xR, extra xr  (bottom-right + extra right)
#
# Geometry (eta < 0):
#   (i,j) in Omega^-, (i+1,j) in Omega^+, (i,j-1) in Omega^+, (i+2,j) in Omega^-.

# %%
s8_pm_ij = s7_pm_ij
s8_pm_i2j = s1_pm_i2j
s8_pp_i1j = {
    xL: -(1 - theta_R) * dx,
    xR: (1 - theta_r) * dx,
    yT: dy,
    yB: -dy,
    x_ext: dx,
    y_ext: -dy,
    uL: uR_p,
    uR: ur_p,
    uT: u[1][1],
    uB: u[1][-1],
    uc: u[1][0],
    u_ext: u[2][-1],
}
s8_pp_ijm1 = {
    xL: -dx,
    xR: dx,
    yT: (1 - theta_B) * dy,
    yB: -dy,
    x_ext: dx,
    y_ext: -dy,
    uL: u[-1][-1],
    uR: u[1][-1],
    uT: uB_p,
    uB: u[0][-2],
    uc: u[0][-1],
    u_ext: u[1][-2],
}

s8_pts_neg = [
    {
        "axis": "y",
        "u_m": uB_m,
        "u_p": uB_p,
        "vars_m": {x: 0, y: -theta_B * dy, **s8_pm_ij},
        "vars_p": {x: 0, y: (1 - theta_B) * dy, **s8_pp_ijm1},
    },
    {
        "axis": "x",
        "u_m": uR_m,
        "u_p": uR_p,
        "vars_m": {x: theta_R * dx, y: 0, **s8_pm_ij},
        "vars_p": {x: -(1 - theta_R) * dx, y: 0, **s8_pp_i1j},
    },
    {
        "axis": "x",
        "u_m": ur_m,
        "u_p": ur_p,
        "vars_m": {x: -theta_r * dx, y: 0, **s8_pm_i2j},
        "vars_p": {x: (1 - theta_r) * dx, y: 0, **s8_pp_i1j},
    },
]

# %%
print("=== Sub-case 8, eta < 0 ===")
M, Nu, d = coeff_case3(-1, s8_pts_neg)

# %%
print("=== Sub-case 8, eta > 0 ===")
M, Nu, d = coeff_case3(+1, points_for_eta_pos(s8_pts_neg))

# %%

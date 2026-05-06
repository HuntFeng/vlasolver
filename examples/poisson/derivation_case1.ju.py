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
# ### Case 1

# %%
import sympy as sp
import sympy.vector as sv
from IPython.display import display
import enum
from typing import Literal

# %%
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


x_i, x_im1, x_ip1 = sp.symbols("x_i x_(i-1) x_(i+1)")
y_j, y_jm1, y_jp1 = sp.symbols("y_j y_(j-1) y_(j+1)")
dx, dy = sp.Symbol(r"\Delta x"), sp.Symbol(r"\Delta y")
xL, xR = sp.symbols("x_L x_R")
yT, yB = sp.symbols("y_T y_B")
# p, pL, pR, pT, pB = sp.symbols(
#     "\\mathbf{p} \\mathbf{p}_L \\mathbf{p}_R \\mathbf{p}_T \\mathbf{p}_B"
# )
x_ext, y_ext = sp.symbols("x_ext y_ext")
theta_L, theta_R, theta_T, theta_B = sp.symbols("theta_L theta_R theta_T theta_B")
coord = sv.CoordSys3D("coord")
x, y = coord.x, coord.y

nx, ny = sp.symbols("n_x n_y")
a, a_tau, b = sp.symbols("a, a_{\\tau}, b")
beta_jump, beta_p, beta_m = sp.symbols("[\\beta], beta^+, beta^-")
# nx, ny = sp.symbols("n_x n_y", cls=sp.Function)
# a, a_tau, b = sp.symbols("a, a_{\\tau}, b", cls=sp.Function)

P_mat = sp.Matrix(
    [
        [x_i**2, x_i * yT, yT**2, x_i, yT, 1],
        [x_i**2, x_i * yB, yB**2, x_i, yB, 1],
        [xL**2, xL * y_j, y_j**2, xL, y_j, 1],
        [xR**2, xR * y_j, y_j**2, xR, y_j, 1],
        [x_i**2, x_i * y_j, y_j**2, x_i, y_j, 1],
        [x_ext**2, x_ext * y_ext, y_ext**2, x_ext, y_ext, 1],
    ]
)
P_mat_inv = P_mat.inv()
P_coeff = P_mat_inv @ sp.Matrix([uT, uB, uL, uR, uc, u_ext])
A, B, C, D, E, F = P_coeff
P = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F


# %%
class Direction(enum.IntFlag):
    R = 1 << 0
    T = 1 << 1
    L = 1 << 2
    B = 1 << 3


def coeff_case1(
    eta: Literal[1] | Literal[-1], dir: Direction, vars_m: dict, vars_p: dict
):
    grad_P = sv.gradient(P)
    dudx = grad_P.components[coord.i]
    dudy = grad_P.components[coord.j]

    if dir == Direction.L or dir == Direction.R:
        # geometric discretization of [\beta u_x]
        if eta < 0:
            beta_ux_jump_geometry = (
                b * nx - beta_jump * ny * (-ny * dudx + nx * dudy) - beta_p * a_tau * ny
            ).subs(vars_m)
        else:
            beta_ux_jump_geometry = (
                b * nx - beta_jump * ny * (-ny * dudx + nx * dudy) - beta_m * a_tau * ny
            ).subs(vars_p)
        # algebraic definition of [\beta u_x]
        beta_ux_jump_algebra = beta_p * dudx.subs(vars_p) - beta_m * dudx.subs(vars_m)
        # equate the two definitions
        equality = beta_ux_jump_algebra - beta_ux_jump_geometry
    else:
        # geometric discretization of [\beta u_y]
        if eta < 0:
            beta_uy_jump_geometry = (
                b * ny + beta_jump * nx * (-ny * dudx + nx * dudy) + beta_p * a_tau * nx
            ).subs(vars_m)
        else:
            beta_uy_jump_geometry = (
                b * ny + beta_jump * nx * (-ny * dudx + nx * dudy) + beta_m * a_tau * nx
            ).subs(vars_p)

        # algebraic definition of [\beta u_y]
        beta_uy_jump_algebra = beta_p * dudy.subs(vars_p) - beta_m * dudy.subs(vars_m)
        # equate the two definitions
        equality = beta_uy_jump_algebra - beta_uy_jump_geometry

    if eta < 0:
        if dir == Direction.R:
            u_m = uR_m
            eq_sub = equality.subs({uR_p: u_m + a}).expand().collect(u_m)
        elif dir == Direction.L:
            u_m = uL_m
            eq_sub = equality.subs({uL_p: u_m + a}).expand().collect(u_m)
        elif dir == Direction.T:
            u_m = uT_m
            eq_sub = equality.subs({uT_p: u_m + a}).expand().collect(u_m)
        elif dir == Direction.B:
            u_m = uB_m
            eq_sub = equality.subs({uB_p: u_m + a}).expand().collect(u_m)

        u_m_coeff = eq_sub.coeff(u_m).simplify()
        u_m_coeff = u_m_coeff.collect([beta_p, beta_m, beta_jump])
        tops = u_m_coeff.as_numer_denom()[0].as_ordered_terms()
        bot = u_m_coeff.as_numer_denom()[1]
        rest = (eq_sub - u_m_coeff * u_m).simplify().expand()
    else:
        if dir == Direction.R:
            u_p = uR_p
            eq_sub = equality.subs({uR_m: u_p - a}).expand().collect(u_p)
        elif dir == Direction.L:
            u_p = uL_p
            eq_sub = equality.subs({uL_m: u_p - a}).expand().collect(u_p)
        elif dir == Direction.T:
            u_p = uT_p
            eq_sub = equality.subs({uT_m: u_p - a}).expand().collect(u_p)
        elif dir == Direction.B:
            u_p = uB_p
            eq_sub = equality.subs({uB_m: u_p - a}).expand().collect(u_p)

        u_p_coeff = eq_sub.coeff(u_p).simplify()
        u_p_coeff = u_p_coeff.collect([beta_p, beta_m, beta_jump])
        tops = u_p_coeff.as_numer_denom()[0].as_ordered_terms()
        bot = u_p_coeff.as_numer_denom()[1]
        rest = (eq_sub - u_p_coeff * u_p).simplify().expand()
    # M
    M = sp.Add(*[(top / bot).cancel().factor() for top in tops])

    # separate terms in 'rest' into those involving u variables and those not
    u_terms = []
    non_u_terms = []
    for term in rest.as_ordered_terms():
        if any(u_var in term.free_symbols for sublist in u for u_var in sublist):
            u_terms.append(term)
        else:
            non_u_terms.append(term)

    d = -sp.Add(
        *[
            term.cancel().factor()
            for term in sp.Add(*non_u_terms).collect([a, b, a_tau]).as_ordered_terms()
        ]
    )
    Nu = -sp.Add(
        *[
            term.cancel().factor()
            for term in sp.Add(*u_terms).collect(flatten(u)).as_ordered_terms()
        ]
    )

    if dir == Direction.L or dir == Direction.R:
        h = dx
    else:
        h = dy
    # print("dudx_p")
    # display(
    #     sp.Add(*[term.cancel().factor()
    #     for term in dudx.subs(vars_p).collect(flatten(u)).as_ordered_terms()]))
    # print("dudx_m")
    # display(sp.Add(*[term.cancel().factor()
    #     for term in dudx.subs(vars_m).collect(flatten(u)).as_ordered_terms()]))
    # print("beta_ux_jump_geometry")
    # display(sp.Add(*[term.cancel().factor()
    #     for term in beta_ux_jump_geometry.collect(flatten(u)).as_ordered_terms()]))
    print("M")
    display(sp.Add(*[(term * h).cancel().factor() for term in M.as_ordered_terms()]))
    print("d")
    display(sp.Add(*[(term * h).cancel().factor() for term in d.as_ordered_terms()]))
    print("Nu")
    display(
        sp.Add(
            *[
                (term * h).cancel().factor()
                for term in Nu.expand().collect(flatten(u)).as_ordered_terms()
            ]
        ).collect(flatten(u))
    )
    
def algebraic_jump(
    eta: Literal[1] | Literal[-1], dir: Direction, vars_m: dict, vars_p: dict
):
    grad_P = sv.gradient(P)
    dudx = grad_P.components[coord.i]
    dudy = grad_P.components[coord.j]
    
    if dir == Direction.L or dir == Direction.R:
        # algebraic definition of [\beta u_x]
        beta_ux_jump_algebra = beta_p * dudx.subs(vars_p) - beta_m * dudx.subs(vars_m)
    else:
        # algebraic definition of [\beta u_y]
        beta_uy_jump_algebra = beta_p * dudy.subs(vars_p) - beta_m * dudy.subs(vars_m)
    
    if eta < 0:
        if dir == Direction.R:
            u_m = uR_m
            beta_ux_jump_algebra = beta_ux_jump_algebra.subs({uR_p: u_m + a})
        elif dir == Direction.L:
            u_m = uL_m
            beta_ux_jump_algebra = beta_ux_jump_algebra.subs({uL_p: u_m + a})
        elif dir == Direction.T:
            u_m = uT_m
            beta_uy_jump_algebra = beta_uy_jump_algebra.subs({uT_p: u_m + a})
        elif dir == Direction.B:
            u_m = uB_m
            beta_uy_jump_algebra = beta_uy_jump_algebra.subs({uB_p: u_m + a})
    else:
        if dir == Direction.R:
            u_p = uR_p
            beta_ux_jump_algebra = beta_ux_jump_algebra.subs({uR_m: u_p - a})
        elif dir == Direction.L:
            u_p = uL_p
            beta_ux_jump_algebra = beta_ux_jump_algebra.subs({uL_m: u_p - a})
        elif dir == Direction.T:
            u_p = uT_p
            beta_uy_jump_algebra = beta_uy_jump_algebra.subs({uT_m: u_p - a})
        elif dir == Direction.B:
            u_p = uB_p
            beta_uy_jump_algebra = beta_uy_jump_algebra.subs({uB_m: u_p - a})
    
    # 1. Find B (coeff of uR_m)
    # 2. Finc C (coeffs of u's)
    # 3. Find a-term (rest of the terms)
    if dir == Direction.R or dir == Direction.L:
        expanded = beta_ux_jump_algebra.expand()
    elif dir == Direction.T or dir == Direction.B:
        expanded = beta_uy_jump_algebra.expand()
    u_pm = u_m if eta < 0 else u_p
    u_pm_coeff = expanded.coeff(u_pm)
    collected = u_pm_coeff.collect([beta_p, beta_m])
    B_mat = sp.Add(*[term.factor() for term in collected.as_ordered_terms()])
    print("B matrix")
    display(B_mat) # only u_unknown coeff
    
    a_coeff = expanded.coeff(a).factor()
    print("a term")
    display(a_coeff*a)
    
    rest = (expanded - u_pm_coeff*u_pm - a_coeff*a).expand()
    collected = rest.collect(flatten(u))
    C_mat = sp.Add(*[term.factor() for term in collected.as_ordered_terms()])
    print("C matrix")
    display(C_mat)


def geometric_jump(
    eta: Literal[1] | Literal[-1], dir: Direction, vars_m: dict, vars_p: dict
):
    grad_P = sv.gradient(P)
    dudx = grad_P.components[coord.i]
    dudy = grad_P.components[coord.j]
    
    if dir == Direction.L or dir == Direction.R:
        # geometric discretization of [\beta u_x]
        if eta < 0:
            beta_ux_jump_geometry = (
                b * nx - beta_jump * ny * (-ny * dudx + nx * dudy) - beta_p * a_tau * ny
            ).subs(vars_m)
        else:
            beta_ux_jump_geometry = (
                b * nx - beta_jump * ny * (-ny * dudx + nx * dudy) - beta_m * a_tau * ny
            ).subs(vars_p)
    else:
        # geometric discretization of [\beta u_y]
        if eta < 0:
            beta_uy_jump_geometry = (
                b * ny + beta_jump * nx * (-ny * dudx + nx * dudy) + beta_p * a_tau * nx
            ).subs(vars_m)
        else:
            beta_uy_jump_geometry = (
                b * ny + beta_jump * nx * (-ny * dudx + nx * dudy) + beta_m * a_tau * nx
            ).subs(vars_p)

    if dir == Direction.L or dir == Direction.R:
        expanded = beta_ux_jump_geometry.expand()
    else:
        expanded = beta_uy_jump_geometry.expand()

    if dir == Direction.R:
        u_pm = uR_m if eta < 0 else uR_p
    if dir == Direction.T:
        u_pm = uT_m if eta < 0 else uT_p
    if dir == Direction.L:
        u_pm = uL_m if eta < 0 else uL_p
    if dir == Direction.B:
        u_pm = uB_m if eta < 0 else uB_p
        
    u_pm_coeff = expanded.coeff(u_pm).factor()
    print("u_pm_coeff")
    display(u_pm_coeff)
    ab_term = expanded.coeff(a_tau)*a_tau + expanded.coeff(b)*b
    print("ab_term")
    display(ab_term)
    rest = (expanded - u_pm_coeff*u_pm - ab_term).expand()
    collected = rest.collect(flatten(u))
    grad_term = sp.Add(*[term.factor() for term in collected.as_ordered_terms()])
    print("grad_term")
    display(grad_term)
    
    grad_ops = sp.Matrix([-2*x*ny, x*nx-y*ny, 2*y*nx, -ny, nx, 0]).T
    u_coeff = -beta_jump*ny*grad_ops@P_mat_inv
    for coeff in u_coeff:
        display(coeff.subs(vars_m).expand().factor())


# %% [markdown]
# ### $\eta < 0$ 

# %%
case1_right_vars_m = {
    x: x_i + theta_R * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_m,
    uB: u[0][-1],
    uT: u[0][1],
    u_ext: u[-1][-1],
}

case1_right_vars_p = {
    x: x_i - (1 - theta_R) * dx,
    y: y_j,
    xL: x_i - (1 - theta_R) * dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    y_ext: y_j - dy,
    uc: u[1][0],
    uL: uR_p,
    uR: u[2][0],
    uB: u[1][-1],
    uT: u[1][1],
    u_ext: u[1][-1],
}
coeff_case1(-1, Direction.R, vars_m=case1_right_vars_m, vars_p=case1_right_vars_p)
algebraic_jump(-1, Direction.R, vars_m=case1_right_vars_m, vars_p=case1_right_vars_p)
geometric_jump(-1, Direction.R, vars_m=case1_right_vars_m, vars_p=case1_right_vars_p)

# %%
grad_ops = sp.Matrix([-2*x*ny, x*nx-y*ny, 2*y*nx, -ny, nx, 0]).T
u_coeff = -beta_jump*ny*grad_ops@P_mat_inv
for coeff in u_coeff:
    display(coeff.subs(vars_m).expand().factor())

# %%

# %%

# %%
case1_top_vars_m = {
    x: x_i,
    y: y_j + theta_T * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: u[1][0],
    uB: u[0][-1],
    uT: uT_m,
    u_ext: u[-1][-1],
}

case1_top_vars_p = {
    x: x_i,
    y: y_j - (1 - theta_T) * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - (1 - theta_T) * dy,
    x_ext: x_i - dx,
    y_ext: y_j + dy,
    uc: u[0][1],
    uL: u[-1][1],
    uR: u[1][1],
    uB: uT_p,
    uT: u[0][2],
    u_ext: u[-1][2],
}
coeff_case1(-1, Direction.T, vars_m=case1_top_vars_m, vars_p=case1_top_vars_p)
algebraic_jump(-1, Direction.T, vars_m=case1_top_vars_m, vars_p=case1_top_vars_p)

# %%
case1_left_vars_m = {
    x: x_i - theta_L * dx,
    y: y_j,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    # y_ext: y_j + dy,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: uL_m,
    uR: u[1][0],
    uB: u[0][-1],
    uT: u[0][1],
    # u_ext: u[1][1],
    u_ext: u[1][-1],
}

case1_left_vars_p = {
    x: x_i + (1 - theta_L) * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + (1 - theta_L) * dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    # y_ext: y_j + dy,
    y_ext: y_j - dy,
    uc: u[-1][0],
    uL: u[-2][0],
    uR: uL_p,
    uB: u[-1][-1],
    uT: u[-1][1],
    # u_ext: u[-2][1],
    u_ext: u[-2][-1],
}

coeff_case1(-1, Direction.L, vars_m=case1_left_vars_m, vars_p=case1_left_vars_p)
algebraic_jump(-1, Direction.L, vars_m=case1_left_vars_m, vars_p=case1_left_vars_p)

# %%
case1_bottom_vars_m = {
    x: x_i,
    y: y_j - theta_B * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    # x_ext: x_i + dx,
    x_ext: x_i - dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: u[1][0],
    uB: uB_m,
    uT: u[0][1],
    # u_ext: u[1][1],
    u_ext: u[-1][1],
}

case1_bottom_vars_p = {
    x: x_i,
    y: y_j + (1 - theta_B) * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + (1 - theta_B) * dy,
    yB: y_j - dy,
    # x_ext: x_i + dx,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][-1],
    uL: u[-1][-1],
    uR: u[1][-1],
    uB: u[0][-2],
    uT: uB_p,
    # u_ext: u[1][-1],
    u_ext: u[-1][-1],
}

coeff_case1(-1, Direction.B, vars_m=case1_bottom_vars_m, vars_p=case1_bottom_vars_p)
algebraic_jump(-1, Direction.B, vars_m=case1_bottom_vars_m, vars_p=case1_bottom_vars_p)

# %% [markdown]
# ### $\eta > 0$

# %%
case1_right_vars_p = {
    x: x_i + theta_R * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_p,
    uB: u[0][-1],
    uT: u[0][1],
    u_ext: u[-1][-1],
}

case1_right_vars_m = {
    x: x_i - (1 - theta_R) * dx,
    y: y_j,
    xL: x_i - (1 - theta_R) * dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    y_ext: y_j - dy,
    uc: u[1][0],
    uL: uR_m,
    uR: u[2][0],
    uB: u[1][-1],
    uT: u[1][1],
    u_ext: u[1][-1],
}
coeff_case1(1, Direction.R, vars_m=case1_right_vars_m, vars_p=case1_right_vars_p)
algebraic_jump(1, Direction.R, vars_m=case1_right_vars_m, vars_p=case1_right_vars_p)

# %%
case1_top_vars_p = {
    x: x_i,
    y: y_j + theta_T * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: u[1][0],
    uB: u[0][-1],
    uT: uT_p,
    u_ext: u[-1][-1],
}

case1_top_vars_m = {
    x: x_i,
    y: y_j - (1 - theta_T) * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - (1 - theta_T) * dy,
    x_ext: x_i - dx,
    y_ext: y_j + dy,
    uc: u[0][1],
    uL: u[-1][1],
    uR: u[1][1],
    uB: uT_m,
    uT: u[0][2],
    u_ext: u[-1][2],
}
coeff_case1(1, Direction.T, vars_m=case1_top_vars_m, vars_p=case1_top_vars_p)
algebraic_jump(1, Direction.T, vars_m=case1_top_vars_m, vars_p=case1_top_vars_p)

# %%
case1_left_vars_p = {
    x: x_i - theta_L * dx,
    y: y_j,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: uL_p,
    uR: u[1][0],
    uB: u[0][-1],
    uT: u[0][1],
    u_ext: u[1][1],
}

case1_left_vars_m = {
    x: x_i + (1 - theta_L) * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + (1 - theta_L) * dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j + dy,
    uc: u[-1][0],
    uL: u[-2][0],
    uR: uL_m,
    uB: u[-1][-1],
    uT: u[-1][1],
    u_ext: u[-2][1],
}

coeff_case1(1, Direction.L, vars_m=case1_left_vars_m, vars_p=case1_left_vars_p)
algebraic_jump(1, Direction.L, vars_m=case1_left_vars_m, vars_p=case1_left_vars_p)

# %%
case1_bottom_vars_p = {
    x: x_i,
    y: y_j - theta_B * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i + dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: u[1][0],
    uB: uB_p,
    uT: u[0][1],
    u_ext: u[1][1],
}

case1_bottom_vars_m = {
    x: x_i,
    y: y_j + (1 - theta_B) * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + (1 - theta_B) * dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    y_ext: y_j - dy,
    uc: u[0][-1],
    uL: u[-1][-1],
    uR: u[1][-1],
    uB: u[0][-2],
    uT: uB_m,
    u_ext: u[1][-1],
}

coeff_case1(1, Direction.B, vars_m=case1_bottom_vars_m, vars_p=case1_bottom_vars_p)
algebraic_jump(1, Direction.B, vars_m=case1_bottom_vars_m, vars_p=case1_bottom_vars_p)

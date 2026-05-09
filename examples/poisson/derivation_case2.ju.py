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
# ### Case 2

# %% jupyter={"source_hidden": true}
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
        [xR**2, xR * y_j, y_j**2, xR, y_j, 1],
        [xL**2, xL * y_j, y_j**2, xL, y_j, 1],
        [x_i**2, x_i * yT, yT**2, x_i, yT, 1],
        [x_i**2, x_i * yB, yB**2, x_i, yB, 1],
        [x_i**2, x_i * y_j, y_j**2, x_i, y_j, 1],
        [x_ext**2, x_ext * y_ext, y_ext**2, x_ext, y_ext, 1],
    ]
)
P_mat_inv = P_mat.inv()
P_coeff = P_mat_inv @ sp.Matrix([uR, uL, uT, uB, uc, u_ext])
A, B, C, D, E, F = P_coeff
P = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F


# %%
class Direction(enum.IntFlag):
    R = 1 << 0
    T = 1 << 1
    L = 1 << 2
    B = 1 << 3


def coeff_case2(
    eta: Literal[1, -1],
    direction: int,
    vars_x_p: dict,
    vars_x_m: dict,
    vars_y_p: dict,
    vars_y_m: dict,
):

    grad_P = sv.gradient(P)
    dudx = grad_P.components[coord.i]
    dudy = grad_P.components[coord.j]

    if eta < 0:
        beta_ux_jump_geometry = (
            b * nx - beta_jump * ny * (-ny * dudx + nx * dudy) - beta_p * a_tau * ny
        ).subs(vars_x_m)
    else:
        beta_ux_jump_geometry = (
            b * nx - beta_jump * ny * (-ny * dudx + nx * dudy) - beta_m * a_tau * ny
        ).subs(vars_x_p)
    beta_ux_jump_algebra = beta_p * dudx.subs(vars_x_p) - beta_m * dudx.subs(vars_x_m)
    equality_x = beta_ux_jump_algebra - beta_ux_jump_geometry
    equality_x *= dx

    if eta < 0:
        beta_uy_jump_geometry = (
            b * ny + beta_jump * nx * (-ny * dudx + nx * dudy) + beta_p * a_tau * nx
        ).subs(vars_y_m)
    else:
        beta_uy_jump_geometry = (
            b * ny + beta_jump * nx * (-ny * dudx + nx * dudy) + beta_m * a_tau * nx
        ).subs(vars_y_p)
    beta_uy_jump_algebra = beta_p * dudy.subs(vars_y_p) - beta_m * dudy.subs(vars_y_m)
    equality_y = beta_uy_jump_algebra - beta_uy_jump_geometry
    equality_y *= dy

    # M
    M = sp.Matrix([[0, 0], [0, 0]])
    Nu = sp.Matrix([0, 0])
    d = sp.Matrix([0, 0])
    # match statement doesn't work since | in case is not bit-wise or operator
    if eta < 0.0:
        if direction == Direction.R | Direction.T:
            ux_m = uR_m
            uy_m = uT_m
            eq_sub_x = equality_x.subs({uR_p: ux_m + a, uT_p: uy_m + a}).expand()
            eq_sub_y = equality_y.subs({uR_p: ux_m + a, uT_p: uy_m + a}).expand()
        elif direction == Direction.L | Direction.T:
            ux_m = uL_m
            uy_m = uT_m
            eq_sub_x = equality_x.subs({uL_p: ux_m + a, uT_p: uy_m + a}).expand()
            eq_sub_y = equality_y.subs({uL_p: ux_m + a, uT_p: uy_m + a}).expand()
        elif direction == Direction.R | Direction.B:
            ux_m = uR_m
            uy_m = uB_m
            eq_sub_x = equality_x.subs({uR_p: ux_m + a, uB_p: uy_m + a}).expand()
            eq_sub_y = equality_y.subs({uR_p: ux_m + a, uB_p: uy_m + a}).expand()
        elif direction == Direction.L | Direction.B:
            ux_m = uL_m
            uy_m = uB_m
            eq_sub_x = equality_x.subs({uL_p: ux_m + a, uB_p: uy_m + a}).expand()
            eq_sub_y = equality_y.subs({uL_p: ux_m + a, uB_p: uy_m + a}).expand()
        else:
            raise ValueError("No such direction...", direction)

        ux_coeff_x = (
            eq_sub_x.coeff(ux_m).simplify().collect([beta_p, beta_m, beta_jump])
        )
        uy_coeff_x = (
            eq_sub_x.coeff(uy_m).simplify().collect([beta_p, beta_m, beta_jump])
        )
        rest_x = (eq_sub_x - ux_coeff_x * ux_m - uy_coeff_x * uy_m).simplify().expand()

        ux_coeff_y = (
            eq_sub_y.coeff(ux_m).simplify().collect([beta_p, beta_m, beta_jump])
        )
        uy_coeff_y = (
            eq_sub_y.coeff(uy_m).simplify().collect([beta_p, beta_m, beta_jump])
        )
        rest_y = (eq_sub_y - ux_coeff_y * ux_m - uy_coeff_y * uy_m).simplify().expand()
    else:
        if direction == Direction.R | Direction.T:
            ux_p = uR_p
            uy_p = uT_p
            eq_sub_x = equality_x.subs({uR_m: ux_p - a, uT_m: uy_p - a}).expand()
            eq_sub_y = equality_y.subs({uR_m: ux_p - a, uT_m: uy_p - a}).expand()
        elif direction == Direction.L | Direction.T:
            ux_p = uL_p
            uy_p = uT_p
            eq_sub_x = equality_x.subs({uL_m: ux_p - a, uT_m: uy_p - a}).expand()
            eq_sub_y = equality_y.subs({uL_m: ux_p - a, uT_m: uy_p - a}).expand()
        elif direction == Direction.R | Direction.B:
            ux_p = uR_p
            uy_p = uB_p
            eq_sub_x = equality_x.subs({uR_m: ux_p - a, uB_m: uy_p - a}).expand()
            eq_sub_y = equality_y.subs({uR_m: ux_p - a, uB_m: uy_p - a}).expand()
        elif direction == Direction.L | Direction.B:
            ux_p = uL_p
            uy_p = uB_p
            eq_sub_x = equality_x.subs({uL_m: ux_p - a, uB_m: uy_p - a}).expand()
            eq_sub_y = equality_y.subs({uL_m: ux_p - a, uB_m: uy_p - a}).expand()
        else:
            raise ValueError("No such direction...", direction)

        ux_coeff_x = (
            eq_sub_x.coeff(ux_p).simplify().collect([beta_p, beta_m, beta_jump])
        )
        uy_coeff_x = (
            eq_sub_x.coeff(uy_p).simplify().collect([beta_p, beta_m, beta_jump])
        )
        rest_x = (eq_sub_x - ux_coeff_x * ux_p - uy_coeff_x * uy_p).simplify().expand()

        ux_coeff_y = (
            eq_sub_y.coeff(ux_p).simplify().collect([beta_p, beta_m, beta_jump])
        )
        uy_coeff_y = (
            eq_sub_y.coeff(uy_p).simplify().collect([beta_p, beta_m, beta_jump])
        )
        rest_y = (eq_sub_y - ux_coeff_y * ux_p - uy_coeff_y * uy_p).simplify().expand()

    M[0, 0] = sp.Add(
        *[
            (top / ux_coeff_x.as_numer_denom()[1]).cancel().factor()
            for top in ux_coeff_x.as_numer_denom()[0].as_ordered_terms()
        ]
    )

    M[0, 1] = sp.Add(
        *[
            (top / uy_coeff_x.as_numer_denom()[1]).cancel().factor()
            for top in uy_coeff_x.as_numer_denom()[0].as_ordered_terms()
        ]
    )

    M[1, 0] = sp.Add(
        *[
            (top / ux_coeff_y.as_numer_denom()[1]).cancel().factor()
            for top in ux_coeff_y.as_numer_denom()[0].as_ordered_terms()
        ]
    )

    M[1, 1] = sp.Add(
        *[
            (top / uy_coeff_y.as_numer_denom()[1]).cancel().factor()
            for top in uy_coeff_y.as_numer_denom()[0].as_ordered_terms()
        ]
    )

    u_terms = []
    non_u_terms = []
    for term in rest_x.as_ordered_terms():
        if any(u_var in term.free_symbols for sublist in u for u_var in sublist):
            u_terms.append(term)
        else:
            non_u_terms.append(term)

    d[0] = -sp.Add(
        *[
            term.cancel().factor()
            for term in sp.Add(*non_u_terms).collect([a, b, a_tau]).as_ordered_terms()
        ]
    )
    Nu[0] = -sp.Add(
        *[
            term.cancel().factor()
            for term in sp.Add(*u_terms).collect(flatten(u)).as_ordered_terms()
        ]
    )

    u_terms = []
    non_u_terms = []
    for term in rest_y.as_ordered_terms():
        if any(u_var in term.free_symbols for sublist in u for u_var in sublist):
            u_terms.append(term)
        else:
            non_u_terms.append(term)

    d[1] = -sp.Add(
        *[
            term.cancel().factor()
            for term in sp.Add(*non_u_terms).collect([a, b, a_tau]).as_ordered_terms()
        ]
    )
    Nu[1] = -sp.Add(
        *[
            term.cancel().factor()
            for term in sp.Add(*u_terms).collect(flatten(u)).as_ordered_terms()
        ]
    )
    print("M[0,0]")
    display(M[0, 0])
    print("M[0,1]")
    display(M[0, 1])
    print("M[1,0]")
    display(M[1, 0])
    print("M[1,1]")
    display(M[1, 1])

    print("Nu[0]")
    display(Nu[0])
    print("Nu[1]")
    display(Nu[1])

    print("d[0]")
    display(d[0])
    print("d[1]")
    display(d[1])

def geometric_jump(
    eta: Literal[1, -1],
    direction: int,
    vars_x_p: dict,
    vars_x_m: dict,
    vars_y_p: dict,
    vars_y_m: dict,
):
    grad_P = sv.gradient(P)
    dudx = grad_P.components[coord.i]
    dudy = grad_P.components[coord.j]

    if eta < 0:
        beta_ux_jump_geometry = (
            b * nx - beta_jump * ny * (-ny * dudx + nx * dudy) - beta_p * a_tau * ny
        ).subs(vars_x_m)
    else:
        beta_ux_jump_geometry = (
            b * nx - beta_jump * ny * (-ny * dudx + nx * dudy) - beta_m * a_tau * ny
        ).subs(vars_x_p)
    beta_ux_jump_algebra = beta_p * dudx.subs(vars_x_p) - beta_m * dudx.subs(vars_x_m)
    equality_x = beta_ux_jump_algebra - beta_ux_jump_geometry
    equality_x *= dx

    if eta < 0:
        beta_uy_jump_geometry = (
            b * ny + beta_jump * nx * (-ny * dudx + nx * dudy) + beta_p * a_tau * nx
        ).subs(vars_y_m)
    else:
        beta_uy_jump_geometry = (
            b * ny + beta_jump * nx * (-ny * dudx + nx * dudy) + beta_m * a_tau * nx
        ).subs(vars_y_p)
    beta_uy_jump_algebra = beta_p * dudy.subs(vars_y_p) - beta_m * dudy.subs(vars_y_m)
    
    expanded_x = beta_ux_jump_geometry.expand()
    expanded_y = beta_uy_jump_geometry.expand()

    u_pm_x = 0
    u_pm_y = 0
    if dir == Direction.R | Direction.T:
        u_pm_x = uR_m if eta < 0 else uR_p
        u_pm_y = uT_m if eta < 0 else uT_p
    if dir == Direction.T | Direction.L:
        u_pm_x = uL_m if eta < 0 else uL_p
        u_pm_y = uT_m if eta < 0 else uT_p
    if dir == Direction.L | Direction.B:
        u_pm_x = uL_m if eta < 0 else uL_p
        u_pm_y = uB_m if eta < 0 else uB_p
    if dir == Direction.B | Direction.R:
        u_pm_x = uR_m if eta < 0 else uR_p
        u_pm_y = uB_m if eta < 0 else uB_p


    for i, expanded in enumerate([expanded_x, expanded_y]):
        print(f"{"x" if i==0 else "y"} direction")
        u_pm = u_pm_x if i==0 else u_pm_y
        u_pm_coeff = expanded.coeff(u_pm).factor()
        print("u_p_coeff")
        display(u_pm_coeff)
        ab_term = expanded.coeff(a_tau) * a_tau + expanded.coeff(b) * b
        print("ab_term")
        display(ab_term)
        rest = (expanded - u_pm_coeff * u_pm - ab_term).expand()
        collected = rest.collect(flatten(u))
        grad_term = sp.Add(*[term.factor() for term in collected.as_ordered_terms()])
        print("grad_term")
        display(grad_term)

    grad_ops = sp.Matrix([-2 * x * ny, x * nx - y * ny, 2 * y * nx, -ny, nx, 0]).T
    u_coeff = -beta_jump * ny * grad_ops @ P_mat_inv
    print("eval at x direction, order R,L,T,B,c,ext")
    for coeff in u_coeff:
        display(coeff.subs(vars_x_m).expand().factor())
    print("eval at y direction, order R,L,T,B,c,ext")
    for coeff in u_coeff:
        display(coeff.subs(vars_y_m).expand().factor())

# %% [markdown]
# ## $\eta < 0$

# %%
case2_right_vars_m = {  # top right
    x: x_i + theta_R * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_m,
    uB: u[0][-1],
    uT: uT_m,
    u_ext: u[-1][-1],
}

case2_top_vars_m = {
    x: x_i,
    y: y_j + theta_T * dy,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_m,
    uB: u[0][-1],
    uT: uT_m,
    u_ext: u[-1][-1],
}

case2_right_vars_p = {
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

case2_top_vars_p = {
    x: x_i,
    y: y_j - (1 - theta_T) * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - (1 - theta_T) * dy,
    x_ext: x_i + dx,
    y_ext: y_j + dy,
    uc: u[0][1],
    uL: u[-1][1],
    uR: u[1][1],
    uB: uT_p,
    uT: u[0][2],
    u_ext: u[2][2],
}

coeff_case2(
    -1,
    Direction.R | Direction.T,
    vars_x_p=case2_right_vars_p,
    vars_x_m=case2_right_vars_m,
    vars_y_p=case2_top_vars_p,
    vars_y_m=case2_top_vars_m,
)
geometric_jump(
    -1,
    Direction.R | Direction.T,
    vars_x_p=case2_right_vars_p,
    vars_x_m=case2_right_vars_m,
    vars_y_p=case2_top_vars_p,
    vars_y_m=case2_top_vars_m,
)

# %%
case2_left_vars_m = {  # left top
    x: x_i - theta_L * dx,
    y: y_j,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: uL_m,
    uR: u[1][0],
    uB: u[0][-1],
    uT: uT_m,
    u_ext: u[1][-1],
}

case2_top_vars_m = {
    x: x_i,
    y: y_j + theta_T * dy,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: uL_m,
    uR: u[1][0],
    uB: u[0][-1],
    uT: uT_m,
    u_ext: u[1][-1],
}

case2_left_vars_p = {
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
    uR: uL_p,
    uB: u[-1][-1],
    uT: u[-1][1],
    u_ext: u[-2][1],
}

case2_top_vars_p = {
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

coeff_case2(
    -1,
    Direction.L | Direction.T,
    vars_x_p=case2_left_vars_p,
    vars_x_m=case2_left_vars_m,
    vars_y_p=case2_top_vars_p,
    vars_y_m=case2_top_vars_m,
)

# %% jupyter={"source_hidden": true}
case2_left_vars_m = {  # left bottom
    x: x_i - theta_L * dx,
    y: y_j,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i + dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: uL_m,
    uR: u[1][0],
    uB: uB_m,
    uT: u[0][1],
    u_ext: u[1][1],
}

case2_bot_vars_m = {
    x: x_i,
    y: y_j - theta_B * dy,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i + dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: uL_m,
    uR: u[1][0],
    uB: uB_m,
    uT: u[0][1],
    u_ext: u[1][1],
}

case2_left_vars_p = {
    x: x_i + (1 - theta_L) * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + (1 - theta_L) * dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[-1][0],
    uL: u[-2][0],
    uR: uL_p,
    uB: u[-1][-1],
    uT: u[-1][1],
    u_ext: u[-2][-1],
}

case2_bot_vars_p = {
    x: x_i,
    y: y_j + (1 - theta_B) * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + (1 - theta_B) * dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][-1],
    uL: u[-1][-1],
    uR: u[1][-1],
    uB: u[1][-2],
    uT: uB_p,
    u_ext: u[-2][-2],
}

coeff_case2(
    -1,
    Direction.L | Direction.B,
    vars_x_p=case2_left_vars_p,
    vars_x_m=case2_left_vars_m,
    vars_y_p=case2_bot_vars_p,
    vars_y_m=case2_bot_vars_m,
)

# %% jupyter={"source_hidden": true}
case2_right_vars_m = {  # right bottom
    x: x_i + theta_R * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i - dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_m,
    uB: uB_m,
    uT: u[0][1],
    u_ext: u[-1][1],
}

case2_bot_vars_m = {
    x: x_i,
    y: y_j - theta_B * dy,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i - dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_m,
    uB: uB_m,
    uT: u[0][1],
    u_ext: u[-1][1],
}

case2_right_vars_p = {
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
    u_ext: u[2][-1],
}

case2_bot_vars_p = {
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
    uB: u[1][-2],
    uT: uB_p,
    u_ext: u[1][-2],
}

coeff_case2(
    -1,
    Direction.R | Direction.B,
    vars_x_p=case2_right_vars_p,
    vars_x_m=case2_right_vars_m,
    vars_y_p=case2_bot_vars_p,
    vars_y_m=case2_bot_vars_m,
)

# %% [markdown]
# ## $\eta > 0$

# %%
case2_right_vars_p = {  # top right
    x: x_i + theta_R * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_p,
    uB: u[0][-1],
    uT: uT_p,
    u_ext: u[-1][-1],
}

case2_top_vars_p = {
    x: x_i,
    y: y_j + theta_T * dy,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_p,
    uB: u[0][-1],
    uT: uT_p,
    u_ext: u[-1][-1],
}

case2_right_vars_m = {
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

case2_top_vars_m = {
    x: x_i,
    y: y_j - (1 - theta_T) * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - (1 - theta_T) * dy,
    x_ext: x_i + dx,
    y_ext: y_j + dy,
    uc: u[0][1],
    uL: u[-1][1],
    uR: u[1][1],
    uB: uT_m,
    uT: u[0][2],
    u_ext: u[2][2],
}

# coeff_case2(
#     1,
#     Direction.R | Direction.T,
#     vars_x_p=case2_right_vars_p,
#     vars_x_m=case2_right_vars_m,
#     vars_y_p=case2_top_vars_p,
#     vars_y_m=case2_top_vars_m,
# )

geometric_jump(
    -1,
    Direction.R | Direction.T,
    vars_x_p=case2_right_vars_p,
    vars_x_m=case2_right_vars_m,
    vars_y_p=case2_top_vars_p,
    vars_y_m=case2_top_vars_m,
)

# %% jupyter={"source_hidden": true}
case2_left_vars_p = {  # left top
    x: x_i - theta_L * dx,
    y: y_j,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: uL_p,
    uR: u[1][0],
    uB: u[0][-1],
    uT: uT_p,
    u_ext: u[1][-1],
}

case2_top_vars_p = {
    x: x_i,
    y: y_j + theta_T * dy,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + theta_T * dy,
    yB: y_j - dy,
    x_ext: x_i + dx,
    y_ext: y_j - dy,
    uc: u[0][0],
    uL: uL_p,
    uR: u[1][0],
    uB: u[0][-1],
    uT: uT_p,
    u_ext: u[1][-1],
}

case2_left_vars_m = {
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

case2_top_vars_m = {
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

coeff_case2(
    1,
    Direction.L | Direction.T,
    vars_x_p=case2_left_vars_p,
    vars_x_m=case2_left_vars_m,
    vars_y_p=case2_top_vars_p,
    vars_y_m=case2_top_vars_m,
)

# %% jupyter={"source_hidden": true}
case2_left_vars_p = {  # left bottom
    x: x_i - theta_L * dx,
    y: y_j,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i + dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: uL_p,
    uR: u[1][0],
    uB: uB_p,
    uT: u[0][1],
    u_ext: u[1][1],
}

case2_bot_vars_p = {
    x: x_i,
    y: y_j - theta_B * dy,
    xL: x_i - theta_L * dx,
    xR: x_i + dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i + dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: uL_p,
    uR: u[1][0],
    uB: uB_p,
    uT: u[0][1],
    u_ext: u[1][1],
}

case2_left_vars_m = {
    x: x_i + (1 - theta_L) * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + (1 - theta_L) * dx,
    yT: y_j + dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[-1][0],
    uL: u[-2][0],
    uR: uL_m,
    uB: u[-1][-1],
    uT: u[-1][1],
    u_ext: u[-2][-1],
}

case2_bot_vars_m = {
    x: x_i,
    y: y_j + (1 - theta_B) * dy,
    xL: x_i - dx,
    xR: x_i + dx,
    yT: y_j + (1 - theta_B) * dy,
    yB: y_j - dy,
    x_ext: x_i - dx,
    y_ext: y_j - dy,
    uc: u[0][-1],
    uL: u[-1][-1],
    uR: u[1][-1],
    uB: u[1][-2],
    uT: uB_m,
    u_ext: u[-2][-2],
}

coeff_case2(
    1,
    Direction.L | Direction.B,
    vars_x_p=case2_left_vars_p,
    vars_x_m=case2_left_vars_m,
    vars_y_p=case2_bot_vars_p,
    vars_y_m=case2_bot_vars_m,
)

# %% jupyter={"source_hidden": true}
case2_right_vars_p = {  # right bottom
    x: x_i + theta_R * dx,
    y: y_j,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i - dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_p,
    uB: uB_p,
    uT: u[0][1],
    u_ext: u[-1][1],
}

case2_bot_vars_p = {
    x: x_i,
    y: y_j - theta_B * dy,
    xL: x_i - dx,
    xR: x_i + theta_R * dx,
    yT: y_j + dy,
    yB: y_j - theta_B * dy,
    x_ext: x_i - dx,
    y_ext: y_j + dy,
    uc: u[0][0],
    uL: u[-1][0],
    uR: uR_p,
    uB: uB_p,
    uT: u[0][1],
    u_ext: u[-1][1],
}

case2_right_vars_m = {
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
    u_ext: u[2][-1],
}

case2_bot_vars_m = {
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
    uB: u[1][-2],
    uT: uB_m,
    u_ext: u[1][-2],
}

coeff_case2(
    1,
    Direction.R | Direction.B,
    vars_x_p=case2_right_vars_p,
    vars_x_m=case2_right_vars_m,
    vars_y_p=case2_bot_vars_p,
    vars_y_m=case2_bot_vars_m,
)

# %%

import enum

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass, field
from collections.abc import Callable

np.set_printoptions(legacy="1.25")  # no type info when printing

EPS = 0.01


@dataclass
class InterCaseResult:
    M_inv_d: np.ndarray
    M_inv_N: np.ndarray
    theta_r: float
    theta_l: float
    theta_t: float
    theta_b: float
    bot_x: float
    bot_y: float
    eps_r: float
    eps_l: float
    eps_t: float
    eps_b: float
    offset: Callable[[int, int], int]
    eps: list = field(default_factory=list)
    theta: list = field(default_factory=list)
    is_x: list = field(default_factory=list)
    dir: list = field(default_factory=list)

class Direction(enum.IntFlag):
    R = 1 << 0  # 0001
    T = 1 << 1  # 0010
    L = 1 << 2  # 0100
    B = 1 << 3  # 1000


def dirsign(direction: int, is_extra=False):
    if direction == Direction.R or direction == Direction.T:
        return 1.0 if not is_extra else -1.0
    elif direction == Direction.L or direction == direction.B:
        return -1.0 if not is_extra else 1.0
    else:
        raise ValueError("Invalid direction for dirsign", direction)


def surface(x: float, y: float) -> float:
    pass


def compute_normal_field() -> tuple[np.ndarray, np.ndarray]:
    n1 = np.zeros((nx, ny))
    n2 = np.zeros((nx, ny))
    for i in range(3, nx - 3):
        for j in range(3, ny - 3):
            x, y = center(i, j)
            dx_eta = (
                -surface(x + 2 * dx, y)
                + 8 * surface(x + dx, y)
                - 8 * surface(x - dx, y)
                + surface(x - 2 * dx, y)
            ) / (12 * dx)
            dy_eta = (
                -surface(x, y + 2 * dy)
                + 8 * surface(x, y + dy)
                - 8 * surface(x, y - dy)
                + surface(x, y - 2 * dy)
            ) / (12 * dy)
            norm = np.sqrt(dx_eta**2 + dy_eta**2)
            if np.isclose(norm, 0.0):
                n1[i, j] = 0.0
                n2[i, j] = 0.0
            else:
                n1[i, j] = dx_eta / norm
                n2[i, j] = dy_eta / norm
    return n1, n2


def index(i: int, j: int) -> int:
    """flatten index"""
    return i * ny + j


def center(i: int, j: int) -> tuple[float, float]:
    return x[i], y[j]


def compute_theta(direction: int, i: int, j: int) -> float:
    x, y = center(i, j)
    eta = surface(x, y)
    is_x_dir = direction in (Direction.R, Direction.L)
    if is_x_dir:
        eta_r = surface(x + dx, y)
        eta_l = surface(x - dx, y)
        d_eta = (eta_r - eta_l) / 2
        dd_eta = (eta_r - 2 * eta + eta_l) / 2
    else:
        eta_t = surface(x, y + dy)
        eta_b = surface(x, y - dy)
        d_eta = (eta_t - eta_b) / 2
        dd_eta = (eta_t - 2 * eta + eta_b) / 2
    if np.isclose(dd_eta, 0.0):
        theta = np.abs(eta / d_eta)
    else:
        s = dirsign(direction)
        theta = (
            -s * d_eta - np.sign(eta) * np.sqrt(d_eta**2 - 4 * dd_eta * eta)
        ) / (2 * dd_eta)
    if theta < 1e-6 or theta > 1.0 - 1e-6:
        theta = 1.0
    return theta


def interp(direction: int, theta: float, i: int, j: int, field: np.ndarray) -> float:
    """cubic interpolation"""
    t_matrix = np.array([1, theta, theta**2, theta**3])
    c_matrix = np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [2.0, -5.0, 4.0, -1.0],
            [-1.0, 3.0, -3.0, 1.0],
        ]
    )
    is_x_dir = direction in (Direction.R, Direction.L)
    s = dirsign(direction)
    if is_x_dir:
        k = i
        points = field[k - 1 : k + 3, j] if s > 0 else field[k - 2 : k + 2, j][::-1]
    else:
        k = j
        points = field[i, k - 1 : k + 3] if s > 0 else field[i, k - 2 : k + 2][::-1]
    return 0.5 * t_matrix @ c_matrix @ points


def compute_P_inv(
    x: float,
    y: float,
    x_r: float,
    x_l: float,
    x_ext: float,
    y_t: float,
    y_b: float,
    y_ext: float,
):
    P_mat = np.array(
        [
            [x_r**2, x_r * y, y**2, x_r, y, 1],  # R
            [x_l**2, x_l * y, y**2, x_l, y, 1],  # L
            [x**2, x * y_t, y_t**2, x, y_t, 1],  # T
            [x**2, x * y_b, y_b**2, x, y_b, 1],  # B
            [x**2, x * y, y**2, x, y, 1],  # ij
            [x_ext**2, x_ext * y_ext, y_ext**2, x_ext, y_ext, 1],  # ext
        ]
    )
    return np.linalg.inv(P_mat)


def compute_a_tau_field() -> np.ndarray:
    """Compute tangential derivative of jump condition a at (i, j)"""
    a_tau = np.zeros((nx, ny))
    for i in range(3, nx - 3):
        for j in range(3, ny - 3):
            dx_a = (-a[i + 2, j] + 8 * a[i + 1, j] - 8 * a[i - 1, j] + a[i - 2, j]) / (
                12 * dx
            )
            dy_a = (-a[i, j + 2] + 8 * a[i, j + 1] - 8 * a[i, j - 1] + a[i, j - 2]) / (
                12 * dy
            )
            a_tau[i, j] = -dx_a * n2[i, j] + dy_a * n1[i, j]
    return a_tau


def coeff_case0(i: int, j: int) -> None:
    """coeff of u_ij and its neighbors for a normal cell"""
    x, y = center(i, j)
    eps_l = permittivity(x - dx / 2, y)
    eps_r = permittivity(x + dx / 2, y)
    eps_b = permittivity(x, y - dy / 2)
    eps_t = permittivity(x, y + dy / 2)

    row_idx = index(i, j)  # laplacian matrix row index
    rows.extend([row_idx] * 5)
    cols.extend(
        [
            index(i - 1, j),
            index(i + 1, j),
            index(i, j - 1),
            index(i, j + 1),
            index(i, j),
        ]
    )
    vals.extend(
        [
            eps_l / dx**2,  # u_[i-1,j]
            eps_r / dx**2,  # u_[i+1,j]
            eps_b / dy**2,  # u_[i,j-1]
            eps_t / dy**2,  # u_[i,j+1]
            (-(eps_l + eps_r) / dx**2 - (eps_b + eps_t) / dy**2),  # u_[i,j]
        ]
    )


def _per_iface_algebraic(s_eta, s, eps_p, eps_m, theta, a_I):
    """Return (B, C, a_term) for a single uncoupled interface."""
    _phi = (3 - 2 * theta) / ((1 - theta) * (2 - theta))
    _psi = (2 * theta + 1) / (theta * (theta + 1))
    B_val = s_eta * s * (eps_p * _phi + eps_m * _psi)
    C_arr = -s_eta * s * np.array(
        [
            -eps_m * theta / (1 + theta),
            eps_m * (1 + theta) / theta,
            eps_p * (2 - theta) / (1 - theta),
            -eps_p * (1 - theta) / (2 - theta),
        ]
    )
    a_val = -s * a_I * eps_p * _phi
    return B_val, C_arr, a_val


def _assemble_MND(
    n_intf,
    B,
    C,
    a_term,
    grad_coeff,
    a_tau_term,
    b_term,
    dirs,
    offset,
    offset_ext,
    stencil_size,
):
    """Assemble M, N, D for an n_intf x n_intf local system.

    M is the n_intf x n_intf matrix of diagonal B terms minus grad_coeff.
    N is the n_intf x stencil_size matrix of far-neighbour coefficients.
    D is the n_intf x 1 RHS vector from a_tau_term + b_term - a_term.
    """
    D = np.array([a_tau_term[d] + b_term[d] - a_term[d] for d in range(n_intf)])

    if isinstance(B, np.ndarray) and B.ndim == 2:
        M = B.copy()
    else:
        M = np.diag(B)

    _grad_idx = {Direction.R: 0, Direction.L: 1, Direction.T: 2, Direction.B: 3}
    for d in range(n_intf):
        for e in range(n_intf):
            M[d, e] -= grad_coeff[d][_grad_idx[dirs[e]]]

    N = np.zeros((n_intf, stencil_size))
    _dir_step = {
        Direction.R: (1, 0),
        Direction.T: (0, 1),
        Direction.L: (-1, 0),
        Direction.B: (0, -1),
    }
    for d in range(n_intf):
        N[d, offset(1, 0)] = grad_coeff[d][0] if not (Direction.R in dirs) else 0.0
        N[d, offset(-1, 0)] = grad_coeff[d][1] if not (Direction.L in dirs) else 0.0
        N[d, offset(0, 1)] = grad_coeff[d][2] if not (Direction.T in dirs) else 0.0
        N[d, offset(0, -1)] = grad_coeff[d][3] if not (Direction.B in dirs) else 0.0
        N[d, offset(0, 0)] = grad_coeff[d][4]
        N[d, offset(*offset_ext)] = grad_coeff[d][5]

        dx_dir, dy_dir = _dir_step[dirs[d]]
        for k in range(len(C[d])):
            N[d, offset((k - 1) * dx_dir, (k - 1) * dy_dir)] -= C[d][k]

    return M, N, D


def case1(direction: int, i: int, j: int) -> InterCaseResult:
    x, y = center(i, j)
    eta = surface(x, y)
    theta = compute_theta(direction, i, j)
    a_tau_I = interp(direction, theta, i, j, a_tau)
    a_I = interp(direction, theta, i, j, a)
    b_I = interp(direction, theta, i, j, b)
    n1_I = interp(direction, theta, i, j, n1)
    n2_I = interp(direction, theta, i, j, n2)

    s = dirsign(direction)
    is_x_dir = direction in (Direction.R, Direction.L)

    # Theta per direction (only current direction uses actual theta)
    theta_r = theta if direction == Direction.R else 1.0
    theta_l = theta if direction == Direction.L else 1.0
    theta_t = theta if direction == Direction.T else 1.0
    theta_b = theta if direction == Direction.B else 1.0

    # eps_p/m: permittivity on positive/negative side of the interface
    if is_x_dir:
        eps_p = permittivity(x + s * (theta + EPS) * dx, y)
        eps_m = permittivity(x + s * (theta - EPS) * dx, y)
    else:
        eps_p = permittivity(x, y + s * (theta + EPS) * dy)
        eps_m = permittivity(x, y + s * (theta - EPS) * dy)
    eps_jump = eps_p - eps_m
    if eta > 0:
        eps_jump = -eps_jump

    # Geometric correction: extension point per direction
    _ext = {
        Direction.R: (x - dx, y - dy, (-1, -1)),
        Direction.T: (x - dx, y - dy, (-1, -1)),
        Direction.L: (x + dx, y - dy, (1, -1)),
        Direction.B: (x - dx, y + dy, (-1, 1)),
    }
    x_ext, y_ext, offset_ext = _ext[direction]

    x_r = x + theta_r * dx
    x_l = x - theta_l * dx
    y_t = y + theta_t * dy
    y_b = y - theta_b * dy
    bot_x = (theta_r + theta_l) / 2 * dx**2
    bot_y = (theta_t + theta_b) / 2 * dy**2
    eps_r = permittivity(x + theta_r * dx / 2, y)
    eps_l = permittivity(x - theta_l * dx / 2, y)
    eps_t = permittivity(x, y + theta_t * dy / 2)
    eps_b = permittivity(x, y - theta_b * dy / 2)

    # algebraic [eps*u_x] and [eps*u_y] Eq(30)
    s_eta = np.sign(eta)
    B_val, C_arr, a_term = _per_iface_algebraic(s_eta, s, eps_p, eps_m, theta, a_I)

    # geometric [eps*u_x] and [eps*u_y] Eq(34)
    P_inv = compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext)
    x_I = x + s * theta * dx if is_x_dir else x
    y_I = y + s * theta * dy if not is_x_dir else y
    grad_tau = np.array(
        [-2 * x_I * n2_I, x_I * n1_I - y_I * n2_I, 2 * y_I * n1_I, -n2_I, n1_I, 0.0]
    )

    if is_x_dir:
        grad_coeff = -dx * eps_jump * n2_I * grad_tau @ P_inv
        a_tau_term = -dx * a_tau_I * eps_p * n2_I
        b_term = dx * b_I * n1_I
    else:
        grad_coeff = dy * eps_jump * n1_I * grad_tau @ P_inv
        a_tau_term = dy * a_tau_I * eps_p * n1_I
        b_term = dy * b_I * n2_I

    offset = lambda ox, oy: (ox + 2) * 5 + (oy + 2)
    M, N, D = _assemble_MND(
        1, [B_val], [C_arr], [a_term], [grad_coeff], [a_tau_term], [b_term],
        [direction], offset, offset_ext, 25,
    )
    M_inv_d = np.linalg.solve(M, D)[0]
    M_inv_N = np.linalg.solve(M, N)[0]

    dr = dx if is_x_dir else dy
    n_norm = n1_I if is_x_dir else n2_I
    n_tang = -n2_I if is_x_dir else n1_I

    # # --- DEBUG: print all case1 values for comparison with C++ ---
    # print(f"=== CASE1 i={i} j={j} dir={direction} ===")
    # print(f"x={x:.15e} y={y:.15e} eta={float(eta):.15e} theta={float(theta):.15e}")
    # print(f"a_tau_I={float(a_tau_I):.15e} a_I={float(a_I):.15e} b_I={float(b_I):.15e} n1_I={float(n1_I):.15e} n2_I={float(n2_I):.15e}")
    # print(f"s={float(s):.1f} is_x_dir={is_x_dir} s_eta={float(s_eta):.15e}")
    # print(f"theta_r={float(theta_r):.15e} theta_l={float(theta_l):.15e} theta_t={float(theta_t):.15e} theta_b={float(theta_b):.15e}")
    # print(f"eps_p={float(eps_p):.15e} eps_m={float(eps_m):.15e} eps_jump={float(eps_jump):.15e}")
    # print(f"x_ext={float(x_ext):.15e} y_ext={float(y_ext):.15e} offset_ext={offset_ext}")
    # print(f"x_r={float(x_r):.15e} x_l={float(x_l):.15e} y_t={float(y_t):.15e} y_b={float(y_b):.15e}")
    # print(f"bot_x={float(bot_x):.15e} bot_y={float(bot_y):.15e}")
    # print(f"eps_r={float(eps_r):.15e} eps_l={float(eps_l):.15e} eps_t={float(eps_t):.15e} eps_b={float(eps_b):.15e}")
    # print(f"B_val={float(B_val):.15e}")
    # print(f"C_arr={np.array2string(np.asarray(C_arr), precision=15, separator=', ', floatmode='unique')}")
    # print(f"a_term={float(a_term):.15e}")
    # print(f"x_I={float(x_I):.15e} y_I={float(y_I):.15e}")
    # print(f"dr={float(dr):.15e} n_norm={float(n_norm):.15e} n_tang={float(n_tang):.15e}")
    # for kk in range(6):
    #     print(f"grad_coeff[{kk}]={float(grad_coeff[kk]):.15e}")
    # print(f"a_tau_term={float(a_tau_term):.15e} b_term={float(b_term):.15e}")
    # print("P_inv:")
    # for ii in range(6):
    #     for jj in range(6):
    #         print(f"  P_inv[{ii}][{jj}]={float(P_inv[ii,jj]):.15e}")
    # print(f"M[0][0]={float(M[0,0]):.15e}")
    # for kk in range(25):
    #     print(f"N[0][{kk}]={float(N[0,kk]):.15e}")
    # print(f"D[0]={float(D[0]):.15e}")
    # print(f"M_inv_d[0]={float(M_inv_d):.15e}")
    # for kk in range(25):
    #     print(f"M_inv_N[0][{kk}]={float(M_inv_N[kk]):.15e}")
    # print(f"=== END CASE1 i={i} j={j} dir={direction} ===")
    # # --- END DEBUG ---
    #
    return InterCaseResult(
        M_inv_d=M_inv_d, M_inv_N=M_inv_N,
        theta_r=theta_r, theta_l=theta_l, theta_t=theta_t, theta_b=theta_b,
        bot_x=bot_x, bot_y=bot_y,
        eps_r=eps_r, eps_l=eps_l, eps_t=eps_t, eps_b=eps_b,
        offset=offset,
        theta=[],
    )


def coeff_case1(direction: int, i: int, j: int) -> None:
    r = case1(direction, i, j)
    row_idx = index(i, j)
    is_x_dir = direction in (Direction.R, Direction.L)

    # Substitution coefficient: eps in current direction / theta / denominator
    _eps_dir = {
        Direction.R: r.eps_r,
        Direction.L: r.eps_l,
        Direction.T: r.eps_t,
        Direction.B: r.eps_b,
    }
    _theta_dir = {
        Direction.R: r.theta_r,
        Direction.L: r.theta_l,
        Direction.T: r.theta_t,
        Direction.B: r.theta_b,
    }
    sub_coeff = (
        _eps_dir[direction] / _theta_dir[direction] / (r.bot_x if is_x_dir else r.bot_y)
    )
    f[i, j] -= r.M_inv_d * sub_coeff

    # Extra stencil entries for directions orthogonal to the current one
    add_terms = {}
    if not (direction & Direction.R):
        add_terms[(1, 0)] = r.eps_r / r.theta_r / r.bot_x
    if not (direction & Direction.L):
        add_terms[(-1, 0)] = r.eps_l / r.theta_l / r.bot_x
    if not (direction & Direction.T):
        add_terms[(0, 1)] = r.eps_t / r.theta_t / r.bot_y
    if not (direction & Direction.B):
        add_terms[(0, -1)] = r.eps_b / r.theta_b / r.bot_y

    for offset_x in range(-2, 3):
        for offset_y in range(-2, 3):
            value = r.M_inv_N[r.offset(offset_x, offset_y)] * sub_coeff
            if (offset_x, offset_y) == (0, 0):
                value += (
                    -(r.eps_r / r.theta_r + r.eps_l / r.theta_l) / r.bot_x
                    - (r.eps_t / r.theta_t + r.eps_b / r.theta_b) / r.bot_y
                )
            else:
                value += add_terms.get((offset_x, offset_y), 0.0)
            rows.append(row_idx)
            cols.append(index(i + offset_x, j + offset_y))
            vals.append(value)


def case2(direction: int, i: int, j: int, debug=False) -> InterCaseResult:
    x, y = center(i, j)
    eta = surface(x, y)

    offset = lambda ox, oy: (ox + 2) * 5 + (oy + 2)

    theta_r = compute_theta(Direction.R, i, j)
    theta_t = compute_theta(Direction.T, i, j)
    theta_l = compute_theta(Direction.L, i, j)
    theta_b = compute_theta(Direction.B, i, j)

    x_r = x + theta_r * dx
    x_l = x - theta_l * dx
    y_t = y + theta_t * dy
    y_b = y - theta_b * dy

    eps_r = permittivity(x + theta_r * dx / 2, y)
    eps_l = permittivity(x - theta_l * dx / 2, y)
    eps_t = permittivity(x, y + theta_t * dy / 2)
    eps_b = permittivity(x, y - theta_b * dy / 2)
    if direction == Direction.R | Direction.T:
        dir = [Direction.R, Direction.T]
        theta = [theta_r, theta_t]  # (x, y)
        eps = [eps_r, eps_t]
    elif direction == Direction.L | Direction.T:
        dir = [Direction.L, Direction.T]
        theta = [theta_l, theta_t]  # (x, y)
        eps = [eps_l, eps_t]
    elif direction == Direction.L | Direction.B:
        dir = [Direction.L, Direction.B]
        theta = [theta_l, theta_b]  # (x, y)
        eps = [eps_l, eps_b]
    elif direction == Direction.R | Direction.B:
        dir = [Direction.R, Direction.B]
        theta = [theta_r, theta_b]  # (x, y)
        eps = [eps_r, eps_b]

    _ext = {
        Direction.R | Direction.T: (x - dx, y - dy, (-1, -1)),
        Direction.T | Direction.L: (x + dx, y - dy, (1, -1)),
        Direction.L | Direction.B: (x + dx, y + dy, (1, 1)),
        Direction.B | Direction.R: (x - dx, y + dy, (-1, 1)),
    }
    x_ext, y_ext, offset_ext = _ext[direction]

    s_eta = np.sign(eta)
    s = [dirsign(dir[d]) for d in range(2)]

    dr = [dx, dy]
    bot_x = (theta_r + theta_l) / 2 * dx**2
    bot_y = (theta_t + theta_b) / 2 * dy**2

    eps_p = [
        permittivity(x + s[0] * (theta[0] + EPS) * dx, y),
        permittivity(x, y + s[1] * (theta[1] + EPS) * dy),
    ]
    eps_m = [
        permittivity(x + s[0] * (theta[0] - EPS) * dx, y),
        permittivity(x, y + s[1] * (theta[1] - EPS) * dy),
    ]
    eps_jump = np.array([eps_p[d] - eps_m[d] for d in range(2)])
    if eta > 0:
        eps_jump = -eps_jump

    # normal evaluated at x and y
    n1_I = np.array([interp(dir[d], theta[d], i, j, n1) for d in range(2)])
    n2_I = np.array([interp(dir[d], theta[d], i, j, n2) for d in range(2)])

    # a_tau at x and y
    a_tau_I = [interp(dir[d], theta[d], i, j, a_tau) for d in range(2)]

    # jump conditions at x and y
    a_I = [interp(dir[d], theta[d], i, j, a) for d in range(2)]
    b_I = [interp(dir[d], theta[d], i, j, b) for d in range(2)]

    # algebraic [eps*u_x] and [eps*u_y] Eq(30)
    B, C, a_term = [], [], []
    for d in range(2):
        B_d, C_d, a_d = _per_iface_algebraic(
            s_eta, s[d], eps_p[d], eps_m[d], theta[d], a_I[d]
        )
        B.append(B_d)
        C.append(C_d)
        a_term.append(a_d)

    P_inv = compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext)
    x_I = np.array([x + s[0] * theta[0] * dx, x])
    y_I = np.array([y, y + s[1] * theta[1] * dy])

    grad_tau = [
        np.array(
            [
                -2 * x_I[d] * n2_I[d],
                x_I[d] * n1_I[d] - y_I[d] * n2_I[d],
                2 * y_I[d] * n1_I[d],
                -n2_I[d],
                n1_I[d],
                0.0,
            ]
        )
        for d in range(2)
    ]
    n_norm = [n1_I[0], n2_I[1]]
    n_tang = [-n2_I[0], n1_I[1]]
    grad_coeff = [
        dr[d] * eps_jump[d] * n_tang[d] * grad_tau[d] @ P_inv for d in range(2)
    ]
    a_tau_term = [dr[d] * a_tau_I[d] * eps_p[d] * n_tang[d] for d in range(2)]
    b_term = [dr[d] * b_I[d] * n_norm[d] for d in range(2)]

    M, N, D = _assemble_MND(
        2, B, C, a_term, grad_coeff, a_tau_term, b_term, dir, offset, offset_ext, 25,
    )

    # if debug:
    #     print(f"=== Python case2 i={i} j={j} dir={direction} ===")
    #     print(f"x={x} y={y} dx={dx} dy={dy}")
    #     print(f"eta={eta} s_eta={s_eta}")
    #     print(f"theta_r={theta_r} theta_l={theta_l} theta_t={theta_t} theta_b={theta_b}")
    #     print(f"x_r={x_r} x_l={x_l} y_t={y_t} y_b={y_b} x_ext={x_ext} y_ext={y_ext}")
    #     print(f"bot_x={bot_x} bot_y={bot_y}")
    #     print(f"eps_r={eps_r} eps_l={eps_l} eps_t={eps_t} eps_b={eps_b}")
    #     print(f"dir = {dir}")
    #     print(f"theta = {theta}")
    #     print(f"eps = {eps}")
    #     print(f"s = {s}")
    #     print(f"eps_p = {eps_p}  eps_m = {eps_m}")
    #     print(f"eps_jump = {eps_jump}")
    #     print(f"n1_I = {n1_I}  n2_I = {n2_I}")
    #     print(f"a_tau_I = {a_tau_I}  a_I = {a_I}  b_I = {b_I}")
    #     print(f"B = {B}")
    #     print(f"C[0] = {C[0]}")
    #     print(f"C[1] = {C[1]}")
    #     print(f"a_term = {a_term}")
    #     print(f"grad_coeff[0] = {grad_coeff[0]}")
    #     print(f"grad_coeff[1] = {grad_coeff[1]}")
    #     print(f"a_tau_term = {a_tau_term}  b_term = {b_term}")
    #     print(f"M = {M}")
    #     print(f"D = {D}")
    #     print(f"N[0] = {N[0]}")
    #     print(f"N[1] = {N[1]}")
    #     print(f"P_inv = {P_inv}")

    M_inv_d = np.linalg.solve(M, D)
    M_inv_N = np.linalg.solve(M, N)

    return InterCaseResult(
        M_inv_d=M_inv_d, M_inv_N=M_inv_N,
        theta_r=theta_r, theta_l=theta_l, theta_t=theta_t, theta_b=theta_b,
        bot_x=bot_x, bot_y=bot_y,
        eps_r=eps_r, eps_l=eps_l, eps_t=eps_t, eps_b=eps_b,
        offset=offset,
        eps=eps, theta=theta,
    )


def coeff_case2(direction: int, i: int, j: int) -> None:
    """coeff of u_ij and its neighbors for a case 2 cell"""
    r = case2(direction, i, j)
    row_idx = index(i, j)

    sub_coeff = [r.eps[0] / r.theta[0] / r.bot_x, r.eps[1] / r.theta[1] / r.bot_y]
    f[i, j] -= r.M_inv_d[0] * sub_coeff[0] + r.M_inv_d[1] * sub_coeff[1]

    # Extra stencil entries for directions orthogonal to the current one
    add_terms = {}
    if not (direction & Direction.R):
        add_terms[(1, 0)] = r.eps_r / r.theta_r / r.bot_x
    if not (direction & Direction.L):
        add_terms[(-1, 0)] = r.eps_l / r.theta_l / r.bot_x
    if not (direction & Direction.T):
        add_terms[(0, 1)] = r.eps_t / r.theta_t / r.bot_y
    if not (direction & Direction.B):
        add_terms[(0, -1)] = r.eps_b / r.theta_b / r.bot_y

    for offset_x in range(-2, 3):
        for offset_y in range(-2, 3):
            value = (
                r.M_inv_N[0, r.offset(offset_x, offset_y)] * sub_coeff[0]
                + r.M_inv_N[1, r.offset(offset_x, offset_y)] * sub_coeff[1]
            )
            if (offset_x, offset_y) == (0, 0):
                value += (
                    -(r.eps_r / r.theta_r + r.eps_l / r.theta_l) / r.bot_x
                    - (r.eps_t / r.theta_t + r.eps_b / r.theta_b) / r.bot_y
                )
            else:
                value += add_terms.get((offset_x, offset_y), 0.0)
            rows.append(row_idx)
            cols.append(index(i + offset_x, j + offset_y))
            vals.append(value)


# ---------------------------------------------------------------------------
# Case 3
# ---------------------------------------------------------------------------
# Case 3 has the same two cuts at (i,j) as case 2 PLUS an extra cut on one
# of the two outer segments emanating from the case-2 corner. This produces
# a third interface point (xr / xt / xl / xb) and so a 3x3 local system.
#
# Sub-cases (matching examples/poisson/derivation_case3.ju.py):
#   1. R|T with extra xr   on segment [(i+1,j), (i+2,j)]
#   2. R|T with extra xt   on segment [(i,j+1), (i,j+2)]
#   3. L|T with extra xt   on segment [(i,j+1), (i,j+2)]
#   4. L|T with extra xl   on segment [(i-2,j), (i-1,j)]
#   5. L|B with extra xl   on segment [(i-2,j), (i-1,j)]
#   6. L|B with extra xb   on segment [(i,j-2), (i,j-1)]
#   7. R|B with extra xb   on segment [(i,j-2), (i,j-1)]
#   8. R|B with extra xr   on segment [(i+1,j), (i+2,j)]


def case3_extra_dir(direction: int, i: int, j: int) -> int:
    x_, y_ = center(i, j)
    extra = 0
    if (Direction.R & direction) and (
        surface(x_ + dx, y_) * surface(x_ + 2 * dx, y_) < 0
    ):
        extra |= Direction.R
    if (Direction.T & direction) and (
        surface(x_, y_ + dy) * surface(x_, y_ + 2 * dy) < 0
    ):
        extra |= Direction.T
    if (Direction.L & direction) and (
        surface(x_ - dx, y_) * surface(x_ - 2 * dx, y_) < 0
    ):
        extra |= Direction.L
    if (Direction.B & direction) and (
        surface(x_, y_ - dy) * surface(x_, y_ - 2 * dy) < 0
    ):
        extra |= Direction.B
    return extra

def case3(direction: int, extra: int, i: int, j: int) -> InterCaseResult:
    x, y = center(i, j)
    eta = surface(x, y)
    s_eta = np.sign(eta)

    offset = lambda ox, oy: (ox + 3) * 7 + (oy + 3)

    theta_r = compute_theta(Direction.R, i, j)
    theta_t = compute_theta(Direction.T, i, j)
    theta_l = compute_theta(Direction.L, i, j)
    theta_b = compute_theta(Direction.B, i, j)

    bot_x = (theta_r + theta_l) / 2 * dx**2
    bot_y = (theta_t + theta_b) / 2 * dy**2

    theta_rr = theta_tt = theta_ll = theta_bb = 0.0
    if Direction.R & extra:
        theta_rr = compute_theta(Direction.R, i + 1, j)  # within [(i+1,j),(i+2,j)]
    if Direction.T & extra:
        theta_tt = compute_theta(Direction.T, i, j + 1)
    if Direction.L & extra:
        theta_ll = compute_theta(Direction.L, i - 1, j)
    if Direction.B & extra:
        theta_bb = compute_theta(Direction.B, i, j - 1)

    x_r = x + theta_r * dx
    x_l = x - theta_l * dx
    y_t = y + theta_t * dy
    y_b = y - theta_b * dy

    eps_r = permittivity(x + theta_r * dx / 2, y)
    eps_l = permittivity(x - theta_l * dx / 2, y)
    eps_t = permittivity(x, y + theta_t * dy / 2)
    eps_b = permittivity(x, y - theta_b * dy / 2)

    # always put the extra direction last in the list for consistent handling below
    s = [dirsign(d) for d in direction] + [dirsign(extra, is_extra=True)]
    # algebraic [eps*u_x] and [eps*u_y]
    if (direction == Direction.T | Direction.R) and (extra == Direction.R):
        dir = [Direction.T, Direction.R, Direction.R]
        theta = [theta_t, theta_r]
        theta_extra = [0, theta_rr]
        eps = [eps_t, eps_r, 0.0]  # last one not used
        eps_p = [
            permittivity(x, y + s[0] * (theta[0] + EPS) * dy),
            permittivity(x + s[1] * (theta[1] + EPS) * dx, y),
            permittivity(x + 2 * dx + s[2] * (theta_extra[1] + EPS) * dx, y),
        ]
        eps_m = [
            permittivity(x, y + s[0] * (theta[0] - EPS) * dy),
            permittivity(x + s[1] * (theta[1] - EPS) * dx, y),
            permittivity(x + 2 * dx + s[2] * (theta_extra[1] - EPS) * dx, y),
        ]
        x_ext, y_ext = x - dx, y - dy
        offset_ext = (-1, -1)
        x_I = [x, x + s[1] * theta[1] * dx, x + s[2] * (theta_extra[1] - 2) * dx]
        y_I = [y + s[0] * theta[0] * dy, y, y]
    elif (direction == Direction.T | Direction.R) and (extra == Direction.T):
        dir = [Direction.R, Direction.T, Direction.T]
        theta = [theta_r, theta_t]
        theta_extra = [0, theta_tt]
        eps = [eps_r, eps_t, 0.0]
        eps_p = [
            permittivity(x + s[0] * (theta[0] + EPS) * dx, y),
            permittivity(x, y + s[1] * (theta[1] + EPS) * dy),
            permittivity(x, y + 2 * dy + s[2] * (theta_extra[1] + EPS) * dy),
        ]
        eps_m = [
            permittivity(x + s[0] * (theta[0] - EPS) * dx, y),
            permittivity(x, y + s[1] * (theta[1] - EPS) * dy),
            permittivity(x, y + 2 * dy + s[2] * (theta_extra[1] - EPS) * dy),
        ]
        x_ext, y_ext = x - dx, y - dy
        offset_ext = (-1, -1)
        x_I = [x + s[0] * theta[0] * dx, x, x]
        y_I = [y, y + s[1] * theta[1] * dy, y + s[2] * (theta_extra[1] - 2) * dy]
    elif (direction == Direction.T | Direction.L) and (extra == Direction.T):
        dir = [Direction.L, Direction.T, Direction.T]
        theta = [theta_l, theta_t]
        theta_extra = [0, theta_tt]
        eps = [eps_l, eps_t, 0.0]
        eps_p = [
            permittivity(x + s[0] * (theta[0] + EPS) * dx, y),
            permittivity(x, y + s[1] * (theta[1] + EPS) * dy),
            permittivity(x, y + 2 * dy + s[2] * (theta_extra[1] + EPS) * dy),
        ]
        eps_m = [
            permittivity(x + s[0] * (theta[0] - EPS) * dx, y),
            permittivity(x, y + s[1] * (theta[1] - EPS) * dy),
            permittivity(x, y + 2 * dy + s[2] * (theta_extra[1] - EPS) * dy),
        ]
        x_ext, y_ext = x + dx, y - dy
        offset_ext = (1, -1)
        x_I = [x + s[0] * theta[0] * dx, x, x]
        y_I = [y, y + s[1] * theta[1] * dy, y + s[2] * (theta_extra[1] - 2) * dy]
    elif (direction == Direction.T | Direction.L) and (extra == Direction.L):
        dir = [Direction.T, Direction.L, Direction.L]
        theta = [theta_t, theta_l]
        theta_extra = [0, theta_ll]
        eps = [eps_t, eps_l, 0.0]
        eps_p = [
            permittivity(x, y + s[0] * (theta[0] + EPS) * dy),
            permittivity(x + s[1] * (theta[1] + EPS) * dx, y),
            permittivity(x - 2 * dx + s[2] * (theta_extra[1] + EPS) * dx, y),
        ]
        eps_m = [
            permittivity(x, y + s[0] * (theta[0] - EPS) * dy),
            permittivity(x + s[1] * (theta[1] - EPS) * dx, y),
            permittivity(x - 2 * dx + s[2] * (theta_extra[1] - EPS) * dx, y),
        ]
        x_ext, y_ext = x + dx, y - dy
        offset_ext = (1, -1)
        x_I = [x, x + s[1] * theta[1] * dx, x + s[2] * (theta_extra[1] - 2) * dx]
        y_I = [y + s[0] * theta[0] * dy, y, y]
    elif (direction == Direction.L | Direction.B) and (extra == Direction.L):
        dir = [Direction.B, Direction.L, Direction.L]
        theta = [theta_b, theta_l]
        theta_extra = [0, theta_ll]
        eps = [eps_b, eps_l, 0.0]
        eps_p = [
            permittivity(x, y + s[0] * (theta[0] + EPS) * dy),
            permittivity(x + s[1] * (theta[1] + EPS) * dx, y),
            permittivity(x - 2 * dx + s[2] * (theta_extra[1] + EPS) * dx, y),
        ]
        eps_m = [
            permittivity(x, y + s[0] * (theta[0] - EPS) * dy),
            permittivity(x + s[1] * (theta[1] - EPS) * dx, y),
            permittivity(x - 2 * dx + s[2] * (theta_extra[1] - EPS) * dx, y),
        ]
        x_ext, y_ext = x + dx, y + dy
        offset_ext = (1, 1)
        x_I = [x, x + s[1] * theta[1] * dx, x + s[2] * (theta_extra[1] - 2) * dx]
        y_I = [y + s[0] * theta[0] * dy, y, y]
    elif (direction == Direction.L | Direction.B) and (extra == Direction.B):
        dir = [Direction.L, Direction.B, Direction.B]
        theta = [theta_l, theta_b]
        theta_extra = [0, theta_bb]
        eps = [eps_l, eps_b, 0.0]
        eps_p = [
            permittivity(x + s[0] * (theta[0] + EPS) * dx, y),
            permittivity(x, y + s[1] * (theta[1] + EPS) * dy),
            permittivity(x, y - 2 * dy + s[2] * (theta_extra[1] + EPS) * dy),
        ]
        eps_m = [
            permittivity(x + s[0] * (theta[0] - EPS) * dx, y),
            permittivity(x, y + s[1] * (theta[1] - EPS) * dy),
            permittivity(x, y - 2 * dy + s[2] * (theta_extra[1] - EPS) * dy),
        ]
        x_ext, y_ext = x + dx, y + dy
        offset_ext = (1, 1)
        x_I = [x + s[0] * theta[0] * dx, x, x]
        y_I = [y, y + s[1] * theta[1] * dy, y + s[2] * (theta_extra[1] - 2) * dy]
    elif (direction == Direction.R | Direction.B) and (extra == Direction.B):
        dir = [Direction.R, Direction.B, Direction.B]
        theta = [theta_r, theta_b]
        theta_extra = [0, theta_bb]
        eps = [eps_r, eps_b, 0.0]
        eps_p = [
            permittivity(x + s[0] * (theta[0] + EPS) * dx, y),
            permittivity(x, y + s[1] * (theta[1] + EPS) * dy),
            permittivity(x, y - 2 * dy + s[2] * (theta_extra[1] + EPS) * dy),
        ]
        eps_m = [
            permittivity(x + s[0] * (theta[0] - EPS) * dx, y),
            permittivity(x, y + s[1] * (theta[1] - EPS) * dy),
            permittivity(x, y - 2 * dy + s[2] * (theta_extra[1] - EPS) * dy),
        ]
        x_ext, y_ext = x - dx, y + dy
        offset_ext = (-1, 1)
        x_I = [x + s[0] * theta[0] * dx, x, x]
        y_I = [y, y + s[1] * theta[1] * dy, y + s[2] * (theta_extra[1] - 2) * dy]
    elif (direction == Direction.R | Direction.B) and (extra == Direction.R):
        dir = [Direction.B, Direction.R, Direction.R]
        theta = [theta_b, theta_r]
        theta_extra = [0, theta_rr]
        eps = [eps_b, eps_r, 0.0]
        eps_p = [
            permittivity(x, y + s[0] * (theta[0] + EPS) * dy),
            permittivity(x + s[1] * (theta[1] + EPS) * dx, y),
            permittivity(x + 2 * dx + s[2] * (theta_extra[1] + EPS) * dx, y),
        ]
        eps_m = [
            permittivity(x, y + s[0] * (theta[0] - EPS) * dy),
            permittivity(x + s[1] * (theta[1] - EPS) * dx, y),
            permittivity(x + 2 * dx + s[2] * (theta_extra[1] - EPS) * dx, y),
        ]
        x_ext, y_ext = x - dx, y + dy
        offset_ext = (-1, 1)
        x_I = [x, x + s[1] * theta[1] * dx, x + s[2] * (theta_extra[1] - 2) * dx]
        y_I = [y + s[0] * theta[0] * dy, y, y]
    else:
        raise ValueError(
            "Invalid direction/extra combination for case 3", direction, extra
        )

    dr = [dx if dir[d] in (Direction.R, Direction.L) else dy for d in range(3)]
    eps_jump = np.array([eps_p[d] - eps_m[d] for d in range(3)])
    if eta > 0:
        eps_jump = -eps_jump
    n1_I = [interp(dir[d], theta[d], i, j, n1) for d in range(2)]
    n2_I = [interp(dir[d], theta[d], i, j, n2) for d in range(2)]
    a_tau_I = [interp(dir[d], theta[d], i, j, a_tau) for d in range(2)]
    a_I = [interp(dir[d], theta[d], i, j, a) for d in range(2)]
    b_I = [interp(dir[d], theta[d], i, j, b) for d in range(2)]
    if extra == Direction.R:
        n1_I.append(interp(Direction.L, theta_extra[1], i + 2, j, n1))
        n2_I.append(interp(Direction.L, theta_extra[1], i + 2, j, n2))
        a_tau_I.append(interp(Direction.L, theta_extra[1], i + 2, j, a_tau))
        a_I.append(interp(Direction.L, theta_extra[1], i + 2, j, a))
        b_I.append(interp(Direction.L, theta_extra[1], i + 2, j, b))
    elif extra == Direction.T:
        n1_I.append(interp(Direction.B, theta_extra[1], i, j + 2, n1))
        n2_I.append(interp(Direction.B, theta_extra[1], i, j + 2, n2))
        a_tau_I.append(interp(Direction.B, theta_extra[1], i, j + 2, a_tau))
        a_I.append(interp(Direction.B, theta_extra[1], i, j + 2, a))
        b_I.append(interp(Direction.B, theta_extra[1], i, j + 2, b))
    elif extra == Direction.L:
        n1_I.append(interp(Direction.R, theta_extra[1], i - 2, j, n1))
        n2_I.append(interp(Direction.R, theta_extra[1], i - 2, j, n2))
        a_tau_I.append(interp(Direction.R, theta_extra[1], i - 2, j, a_tau))
        a_I.append(interp(Direction.R, theta_extra[1], i - 2, j, a))
        b_I.append(interp(Direction.R, theta_extra[1], i - 2, j, b))
    elif extra == Direction.B:
        n1_I.append(interp(Direction.T, theta_extra[1], i, j - 2, n1))
        n2_I.append(interp(Direction.T, theta_extra[1], i, j - 2, n2))
        a_tau_I.append(interp(Direction.T, theta_extra[1], i, j - 2, a_tau))
        a_I.append(interp(Direction.T, theta_extra[1], i, j - 2, a))
        b_I.append(interp(Direction.T, theta_extra[1], i, j - 2, b))
    else:
        raise ValueError("bad extra direction", extra)

    # algebraic [eps*u_x] and [eps*u_y]
    B = np.zeros((3, 3))
    # the direction with no extra cut
    B[0, 0] = (
        s_eta
        * s[0]
        * (
            eps_p[0]
            * (3 - 2 * theta[0] - theta_extra[0])
            / ((1 - theta[0]) * (2 - theta[0] - theta_extra[0]))
            + eps_m[0] * (2 * theta[0] + 1) / (theta[0] * (theta[0] + 1))
        )
    )
    B[0, 1] = 0.0
    B[0, 2] = 0.0
    # direction with extra cut
    B[1, 0] = 0.0
    B[1, 1] = (
        s_eta
        * s[1]
        * (
            eps_p[1]
            * (3 - 2 * theta[1] - theta_extra[1])
            / ((1 - theta[1]) * (2 - theta[1] - theta_extra[1]))
            + eps_m[1] * (2 * theta[1] + 1) / (theta[1] * (theta[1] + 1))
        )
    )
    B[1, 2] = (
        s_eta
        * s[1]
        * eps_p[1]
        * (1 - theta[1])
        / ((2 - theta[1] - theta_extra[1]) * (1 - theta_extra[1]))
    )
    # extr cut associated
    B[2, 0] = 0.0
    B[2, 1] = (
        s_eta
        * s[2]
        * (
            eps_p[2]
            * (1 - theta_extra[1])
            / ((2 - theta[1] - theta_extra[1]) * (1 - theta[1]))
        )
    )
    B[2, 2] = (
        s_eta
        * s[2]
        * (
            eps_p[2]
            * (3 - 2 * theta_extra[1] - theta[1])
            / ((1 - theta_extra[1]) * (2 - theta[1] - theta_extra[1]))
            + eps_m[2]
            * (2 * theta_extra[1] + 1)
            / (theta_extra[1] * (theta_extra[1] + 1))
        )
    )

    C = np.zeros((3, 5))  # i-1, i, i+1, i+2, i+3
    # direction with no extra cut
    C[0] = (
        -s_eta
        * s[0]
        * np.array(
            [
                -eps_m[0] * theta[0] / (1 + theta[0]),
                eps_m[0] * (1 + theta[0]) / theta[0],
                eps_p[0] * (2 - theta[0]) / (1 - theta[0]),
                -eps_p[0] * (1 - theta[0]) / (2 - theta[0]),
                0,
            ]
        )
    )
    # direction with extra cut
    C[1] = (
        -s_eta
        * s[1]
        * np.array(
            [
                -eps_m[1] * theta[1] / (1 + theta[1]),
                eps_m[1] * (theta[1] + 1) / (theta[1] * (theta[0] + 1)),
                eps_p[1]
                * (2 - theta[1] - theta_extra[1])
                / ((1 - theta[1]) * (1 - theta_extra[1])),
                0,
                0,
            ]
        )
    )
    # extra cut associated
    C[2] = (
        s_eta
        * s[2]
        * np.array(
            [
                0,
                0,
                -eps_p[2]
                * (2 - theta[1] - theta_extra[1])
                / ((2 - theta[1]) * (1 - theta_extra[1])),
                -eps_m[2] * (theta_extra[1] + 1) / theta_extra[1],
                eps_m[2] * theta_extra[1] / (theta_extra[1] + 1),
            ]
        )
    )

    a_term = np.zeros(3)
    a_term[0] = (
        -s[0]
        * a_I[0]
        * eps_p[0]
        * (3 - 2 * theta[0])
        / ((1 - theta[0]) * (2 - theta[0]))
    )
    a_term[1] = (
        -s[1]
        * a_I[1]
        * eps_p[1]
        * (3 - 2 * theta[1] - theta_extra[1])
        / ((1 - theta[1]) * (2 - theta[1] - theta_extra[1]))
    )
    a_term[2] = (
        -s[2]
        * a_I[2]
        * eps_p[1]
        * (3 - 2 * theta_extra[1] - theta[1])
        / ((1 - theta_extra[1]) * (2 - theta[1] - theta_extra[1]))
    )

    n_norm = [
        n1_I[d] if dir[d] in (Direction.R, Direction.L) else n2_I[d] for d in range(3)
    ]
    n_tang = [
        -n2_I[d] if dir[d] in (Direction.R, Direction.L) else n1_I[d] for d in range(3)
    ]

    P_inv = compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext)
    grad_tau = [
        np.array(
            [
                -2 * x_I[d] * n2_I[d],
                x_I[d] * n1_I[d] - y_I[d] * n2_I[d],
                2 * y_I[d] * n1_I[d],
                -n2_I[d],
                n1_I[d],
                0.0,
            ]
        )
        for d in range(3)
    ]
    grad_coeff = [
        dr[d] * eps_jump[d] * n_tang[d] * grad_tau[d] @ P_inv for d in range(3)
    ]
    a_tau_term = [dr[d] * a_tau_I[d] * eps_p[d] * n_tang[d] for d in range(3)]
    b_term = [dr[d] * b_I[d] * n_norm[d] for d in range(3)]

    M, N, D = _assemble_MND(
        3, B, C, a_term, grad_coeff, a_tau_term, b_term, dir, offset, offset_ext, 49,
    )
    M_inv_d = np.linalg.solve(M, D)
    M_inv_N = np.linalg.solve(M, N)

    return InterCaseResult(
        M_inv_d=M_inv_d, M_inv_N=M_inv_N,
        theta_r=theta_r, theta_l=theta_l, theta_t=theta_t, theta_b=theta_b,
        bot_x=bot_x, bot_y=bot_y,
        eps_r=eps_r, eps_l=eps_l, eps_t=eps_t, eps_b=eps_b,
        offset=offset,
        eps=eps, theta=theta,
    )


def coeff_case3(direction: int, extra: int, i: int, j: int) -> None:
    r = case3(direction, extra, i, j)
    row_idx = index(i, j)

    sub_coeff = [r.eps[0] / r.theta[0] / r.bot_x, r.eps[1] / r.theta[1] / r.bot_y]
    f[i, j] -= r.M_inv_d[0] * sub_coeff[0] + r.M_inv_d[1] * sub_coeff[1]

    # Extra stencil entries for directions orthogonal to the current one
    add_terms = {}
    if not (direction & Direction.R):
        add_terms[(1, 0)] = r.eps_r / r.theta_r / r.bot_x
    if not (direction & Direction.L):
        add_terms[(-1, 0)] = r.eps_l / r.theta_l / r.bot_x
    if not (direction & Direction.T):
        add_terms[(0, 1)] = r.eps_t / r.theta_t / r.bot_y
    if not (direction & Direction.B):
        add_terms[(0, -1)] = r.eps_b / r.theta_b / r.bot_y

    for offset_x in range(-3, 4):
        for offset_y in range(-3, 4):
            value = (
                r.M_inv_N[0, r.offset(offset_x, offset_y)] * sub_coeff[0]
                + r.M_inv_N[1, r.offset(offset_x, offset_y)] * sub_coeff[1]
            )
            if (offset_x, offset_y) == (0, 0):
                value += (
                    -(r.eps_r / r.theta_r + r.eps_l / r.theta_l) / r.bot_x
                    - (r.eps_t / r.theta_t + r.eps_b / r.theta_b) / r.bot_y
                )
            else:
                value += add_terms.get((offset_x, offset_y), 0.0)
            rows.append(row_idx)
            cols.append(index(i + offset_x, j + offset_y))
            vals.append(value)



def case4(direction: int, i: int, j: int) -> InterCaseResult:
    x, y = center(i, j)
    eta = surface(x, y)

    offset = lambda ox, oy: (ox + 2) * 5 + (oy + 2)

    theta_r = compute_theta(Direction.R, i, j)
    theta_t = compute_theta(Direction.T, i, j)
    theta_l = compute_theta(Direction.L, i, j)
    theta_b = compute_theta(Direction.B, i, j)

    x_r = x + theta_r * dx
    x_l = x - theta_l * dx
    y_t = y + theta_t * dy
    y_b = y - theta_b * dy

    eps_r = permittivity(x + theta_r * dx / 2, y)
    eps_l = permittivity(x - theta_l * dx / 2, y)
    eps_t = permittivity(x, y + theta_t * dy / 2)
    eps_b = permittivity(x, y - theta_b * dy / 2)
    if direction == Direction.R | Direction.T | Direction.L:
        dir = [Direction.R, Direction.T, Direction.L]
        theta = [theta_r, theta_t, theta_l]
        eps = [eps_r, eps_t, eps_l]
        x_I = [x + theta_r * dx, x, x - theta_l * dx]
        y_I = [y, y + theta_t * dy, y]
    elif direction == Direction.L | Direction.T | Direction.B:
        dir = [Direction.L, Direction.T, Direction.B]
        theta = [theta_l, theta_t, theta_b]
        eps = [eps_l, eps_t, eps_b]
        x_I = [x - theta_l * dx, x, x]
        y_I = [y, y + theta_t * dy, y - theta_b * dy]
    elif direction == Direction.L | Direction.B | Direction.R:
        dir = [Direction.L, Direction.B, Direction.R]
        theta = [theta_l, theta_b, theta_r]
        eps = [eps_l, eps_b, eps_r]
        x_I = [x - theta_l * dx, x, x + theta_r * dx]
        y_I = [y, y - theta_b * dy, y]
    elif direction == Direction.R | Direction.B | Direction.T:
        dir = [Direction.R, Direction.B, Direction.T]
        theta = [theta_r, theta_b, theta_t]
        eps = [eps_r, eps_b, eps_t]
        x_I = [x + theta_r * dx, x, x]
        y_I = [y, y - theta_b * dy, y + theta_t * dy]
    else:
        raise ValueError("Invalid direction for case 4", direction)

    _ext = {
        Direction.R | Direction.T | Direction.L: (x - dx, y - dy, (-1, -1)),
        Direction.R | Direction.T | Direction.B: (x - dx, y + dy, (-1, 1)),
        Direction.R | Direction.B | Direction.L: (x + dx, y + dy, (1, 1)),
        Direction.T | Direction.B | Direction.L: (x + dx, y - dy, (1, -1)),
    }
    x_ext, y_ext, offset_ext = _ext[direction]

    s_eta = np.sign(eta)
    s = [dirsign(dir[d]) for d in range(3)]

    is_x = [d in (Direction.R, Direction.L) for d in dir]
    dr = [dx if is_x[d] else dy for d in range(3)]
    bot_x = (theta_r + theta_l) / 2 * dx**2
    bot_y = (theta_t + theta_b) / 2 * dy**2

    eps_p = []
    eps_m = []
    for d in range(3):
        if is_x[d]:
            eps_p.append(permittivity(x + s[d] * (theta[d] + EPS) * dx, y))
            eps_m.append(permittivity(x + s[d] * (theta[d] - EPS) * dx, y))
        else:
            eps_p.append(permittivity(x, y + s[d] * (theta[d] + EPS) * dy))
            eps_m.append(permittivity(x, y + s[d] * (theta[d] - EPS) * dy))
    eps_jump = np.array([eps_p[d] - eps_m[d] for d in range(3)])
    if eta > 0:
        eps_jump = -eps_jump

    n1_I = np.array([interp(dir[d], theta[d], i, j, n1) for d in range(3)])
    n2_I = np.array([interp(dir[d], theta[d], i, j, n2) for d in range(3)])
    a_tau_I = [interp(dir[d], theta[d], i, j, a_tau) for d in range(3)]
    a_I = [interp(dir[d], theta[d], i, j, a) for d in range(3)]
    b_I = [interp(dir[d], theta[d], i, j, b) for d in range(3)]

    B, C, a_term = [], [], []
    for d in range(3):
        B_d, C_d, a_d = _per_iface_algebraic(
            s_eta, s[d], eps_p[d], eps_m[d], theta[d], a_I[d]
        )
        B.append(B_d)
        C.append(C_d)
        a_term.append(a_d)

    # geometric [eps*u_x] and [eps*u_y]
    P_inv = compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext)
    grad_tau = [
        np.array(
            [
                -2 * x_I[d] * n2_I[d],
                x_I[d] * n1_I[d] - y_I[d] * n2_I[d],
                2 * y_I[d] * n1_I[d],
                -n2_I[d],
                n1_I[d],
                0.0,
            ]
        )
        for d in range(3)
    ]
    n_norm = [n1_I[d] if is_x[d] else n2_I[d] for d in range(3)]
    n_tang = [-n2_I[d] if is_x[d] else n1_I[d] for d in range(3)]
    grad_coeff = [
        dr[d] * eps_jump[d] * n_tang[d] * grad_tau[d] @ P_inv for d in range(3)
    ]
    a_tau_term = [dr[d] * a_tau_I[d] * eps_p[d] * n_tang[d] for d in range(3)]
    b_term = [dr[d] * b_I[d] * n_norm[d] for d in range(3)]

    M, N, D = _assemble_MND(
        3, B, C, a_term, grad_coeff, a_tau_term, b_term, dir, offset, offset_ext, 25,
    )
    M_inv_d = np.linalg.solve(M, D)
    M_inv_N = np.linalg.solve(M, N)

    return InterCaseResult(
        M_inv_d=M_inv_d, M_inv_N=M_inv_N,
        theta_r=theta_r, theta_l=theta_l, theta_t=theta_t, theta_b=theta_b,
        bot_x=bot_x, bot_y=bot_y,
        eps_r=eps_r, eps_l=eps_l, eps_t=eps_t, eps_b=eps_b,
        offset=offset,
        eps=eps, theta=theta, is_x=is_x, dir=dir,
    )


def coeff_case4(direction: int, i: int, j: int) -> None:
    """coeff of u_ij and its neighbors for a case 4 cell (3 cut interfaces)."""
    r = case4(direction, i, j)
    row_idx = index(i, j)

    sub_coeff = [r.eps[d] / r.theta[d] / (r.bot_x if r.is_x[d] else r.bot_y) for d in range(3)]
    f[i, j] -= sum(r.M_inv_d[d] * sub_coeff[d] for d in range(3))

    add_terms = {}
    if not (direction & Direction.R):
        add_terms[(1, 0)] = r.eps_r / r.theta_r / r.bot_x
    if not (direction & Direction.L):
        add_terms[(-1, 0)] = r.eps_l / r.theta_l / r.bot_x
    if not (direction & Direction.T):
        add_terms[(0, 1)] = r.eps_t / r.theta_t / r.bot_y
    if not (direction & Direction.B):
        add_terms[(0, -1)] = r.eps_b / r.theta_b / r.bot_y

    for offset_x in range(-2, 3):
        for offset_y in range(-2, 3):
            value = sum(
                r.M_inv_N[d, r.offset(offset_x, offset_y)] * sub_coeff[d] for d in range(3)
            )
            if (offset_x, offset_y) == (0, 0):
                value += (
                    -(r.eps_r / r.theta_r + r.eps_l / r.theta_l) / r.bot_x
                    - (r.eps_t / r.theta_t + r.eps_b / r.theta_b) / r.bot_y
                )
            else:
                value += add_terms.get((offset_x, offset_y), 0.0)
            rows.append(row_idx)
            cols.append(index(i + offset_x, j + offset_y))
            vals.append(value)


def construct_matrix():
    for i in range(nx):
        for j in range(ny):
            if i < 3 or i >= nx - 3 or j < 3 or j >= ny - 3:
                rows.append(index(i, j))
                cols.append(index(i, j))
                vals.append(1.0)
                f[i, j] = u_exact[i, j]  # dirichlet bc
                continue

            x, y = center(i, j)
            eta = surface(x, y)
            eta_l = surface(x - dx, y)
            eta_r = surface(x + dx, y)
            eta_b = surface(x, y - dy)
            eta_t = surface(x, y + dy)

            direction = 0
            if eta * eta_l < 0:
                direction |= Direction.L
            if eta * eta_r < 0:
                direction |= Direction.R
            if eta * eta_b < 0:
                direction |= Direction.B
            if eta * eta_t < 0:
                direction |= Direction.T

            match direction.bit_count():
                case 0:
                    coeff_case0(i, j)
                case 1:
                    coeff_case1(direction, i, j)
                case 2:
                    extra = case3_extra_dir(direction, i, j)
                    if extra.bit_count() == 0:
                        coeff_case2(direction, i, j)
                    elif extra.bit_count() == 1:
                        breakpoint()
                        coeff_case3(direction, extra, i, j)
                    else:
                        raise ValueError(
                            f"Invalid extra direction {extra} at ({i},{j}), use finer grid."
                        )
                case 3:
                    breakpoint()
                    coeff_case4(direction, i, j)
                case _:
                    raise ValueError("All four sides cut at ({i},{j}), use finer grid.")

    A = coo_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    # convert csr for cg / gmres solve
    # convert to csc for lu solve
    return A.tocsr()


def interface_value_case0(
    i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    return u[i - 1, j], u[i + 1, j], u[i, j - 1], u[i, j + 1], 1.0, 1.0, 1.0, 1.0


def interface_value_case1(
    direction: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    r = case1(direction, i, j)

    all_offsets = [(ox, oy) for ox in range(-2, 3) for oy in range(-2, 3)]
    u_arr = np.array([u[i + di, j + dj] for (di, dj) in all_offsets])
    ghost = float(r.M_inv_N @ u_arr + r.M_inv_d)
    u_l = ghost if direction & Direction.L else u[i - 1, j]
    u_r = ghost if direction & Direction.R else u[i + 1, j]
    u_b = ghost if direction & Direction.B else u[i, j - 1]
    u_t = ghost if direction & Direction.T else u[i, j + 1]
    return u_l, u_r, u_b, u_t, r.theta_l, r.theta_r, r.theta_b, r.theta_t


def interface_value_case2(
    direction: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    r = case2(direction, i, j)

    all_offsets = [(ox, oy) for ox in range(-2, 3) for oy in range(-2, 3)]
    u_arr = np.array([u[i + di, j + dj] for (di, dj) in all_offsets])
    ghosts = r.M_inv_N @ u_arr + r.M_inv_d
    u_l = ghosts[0] if direction & Direction.L else u[i - 1, j]
    u_r = ghosts[0] if direction & Direction.R else u[i + 1, j]
    u_b = ghosts[1] if direction & Direction.B else u[i, j - 1]
    u_t = ghosts[1] if direction & Direction.T else u[i, j + 1]
    return u_l, u_r, u_b, u_t, r.theta_l, r.theta_r, r.theta_b, r.theta_t


def interface_value_case3(
    direction: int, extra: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Reconstruct interface values for a case-3 cell."""
    r = case3(direction, extra, i, j)

    all_offsets = [(ox, oy) for ox in range(-3, 4) for oy in range(-3, 4)]
    u_arr = np.array([u[i + di, j + dj] for (di, dj) in all_offsets])
    ghosts = r.M_inv_N @ u_arr + r.M_inv_d
    u_l = ghosts[0] if direction & Direction.L else u[i - 1, j]
    u_r = ghosts[0] if direction & Direction.R else u[i + 1, j]
    u_b = ghosts[1] if direction & Direction.B else u[i, j - 1]
    u_t = ghosts[1] if direction & Direction.T else u[i, j + 1]
    return u_l, u_r, u_b, u_t, r.theta_l, r.theta_r, r.theta_b, r.theta_t

def interface_value_case4(
    direction: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Reconstruct interface values for a case-4 cell."""
    r = case4(direction, i, j)

    all_offsets = [(ox, oy) for ox in range(-2, 3) for oy in range(-2, 3)]
    u_arr = np.array([u[i + di, j + dj] for (di, dj) in all_offsets])
    ghosts = r.M_inv_N @ u_arr + r.M_inv_d
    u_l = ghosts[r.dir.index(Direction.L)] if direction & Direction.L else u[i - 1, j]
    u_r = ghosts[r.dir.index(Direction.R)] if direction & Direction.R else u[i + 1, j]
    u_b = ghosts[r.dir.index(Direction.B)] if direction & Direction.B else u[i, j - 1]
    u_t = ghosts[r.dir.index(Direction.T)] if direction & Direction.T else u[i, j + 1]
    return u_l, u_r, u_b, u_t, r.theta_l, r.theta_r, r.theta_b, r.theta_t


def gradient(u: np.ndarray):
    """Compute the gradient of u using central difference."""
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)

    # boundaries (not important, will be discarded when computing error)
    dudx[0, :] = (u[1, :] - u[0, :]) / dx
    dudx[-1, :] = (u[-1, :] - u[-2, :]) / dx
    dudy[:, 0] = (u[:, 1] - u[:, 0]) / dy
    dudy[:, -1] = (u[:, -1] - u[:, -2]) / dy

    # near interface
    for i in range(3, nx - 3):
        for j in range(3, ny - 3):
            x, y = center(i, j)
            eta = surface(x, y)
            eta_l = surface(x - dx, y)
            eta_r = surface(x + dx, y)
            eta_b = surface(x, y - dy)
            eta_t = surface(x, y + dy)

            direction = 0
            if eta * eta_l < 0:
                direction |= Direction.L
            if eta * eta_r < 0:
                direction |= Direction.R
            if eta * eta_b < 0:
                direction |= Direction.B
            if eta * eta_t < 0:
                direction |= Direction.T

            match direction.bit_count():
                case 0:
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                        interface_value_case0(i, j, u)
                    )
                case 1:
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                        interface_value_case1(direction, i, j, u)
                    )
                case 2:
                    extra = case3_extra_dir(direction, i, j)
                    if extra.bit_count() == 0:
                        u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                            interface_value_case2(direction, i, j, u)
                        )
                    elif extra.bit_count() == 1:
                        u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                            interface_value_case3(direction, extra, i, j, u)
                        )
                    else:
                        raise ValueError(
                            f"Invalid extra direction {extra} at ({i},{j}), use finer grid."
                        )
                case 3:
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                        interface_value_case4(direction, i, j, u)
                    )
                case _:
                    raise ValueError(
                        f"All four sides cut at one cell {(i, j)}, use finer grid."
                    )

            dudx[i, j] = (
                -(theta_r**2) * u_l
                + (theta_r**2 - theta_l**2) * u[i, j]
                + theta_l**2 * u_r
            ) / (theta_l * theta_r * (theta_l + theta_r) * dx)
            dudy[i, j] = (
                -(theta_t**2) * u_b
                + (theta_t**2 - theta_b**2) * u[i, j]
                + theta_b**2 * u_t
            ) / (theta_b * theta_t * (theta_b + theta_t) * dy)
    return dudx, dudy


def convergence_test1():
    """
    [beta]=-1,
    a(y=0.3)=-exp(-0.09), a(y=0.6)=-exp(-0.36),
    b(y=0.3)=-1.2exp(-0.09), b(y=0.6)=2.4exp(-0.36),
    f=(8y^2-4)exp(-y^2) for 0.3<y<0.6 else 0,
    surface=abs(y-0.45)-0.15
    """
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact
    global a, b, a_tau, n1, n2
    global rows, cols, vals
    global surface, permittivity

    def surface(x, y):
        return np.abs(y - 0.45) - 0.15

    def permittivity(x, y):
        return 2.0 if surface(x, y) <= 0 else 1.0

    n_range = 2 ** np.arange(4, 10, dtype=int)
    errors_u = np.zeros(n_range.size)
    errors_du = np.zeros(n_range.size)

    for i, n in enumerate(n_range):
        nx, ny = 9, n
        dx, dy = 1.0 / nx, 1.0 / ny
        x = np.arange(dx / 2, 1.0 + dx / 2, dx)
        y = np.arange(dy / 2, 1.0 + dy / 2, dy)
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = np.piecewise(
            Y,
            [(Y < 0.6) & (Y > 0.3)],
            [lambda y: (8 * y**2 - 4) * np.exp(-(y**2)), 0.0],
        )
        u_exact = np.piecewise(
            Y,
            [(Y < 0.6) & (Y > 0.3)],
            [lambda y: np.exp(-(y**2)), 0.0],
        )
        dudx_exact = np.zeros_like(X)
        dudy_exact = np.piecewise(
            Y,
            [(Y < 0.6) & (Y > 0.3)],
            [lambda y: -2 * y * np.exp(-(y**2)), 0.0],
        )
        a = np.piecewise(
            Y,
            [Y < 0.45, Y >= 0.45],
            [lambda y: -np.exp(-0.09), lambda y: -np.exp(-0.36)],
        )
        b = np.piecewise(
            Y,
            [Y < 0.45, Y >= 0.45],
            [lambda y: -1.2 * np.exp(-0.09), lambda y: 2.4 * np.exp(-0.36)],
        )
        n1, n2 = compute_normal_field()
        a_tau = compute_a_tau_field()

        rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
        A = construct_matrix()
        u = spsolve(A, f.flatten())
        u = u.reshape((nx, ny))
        dudx, dudy = gradient(u)
        errors_u[i] = np.max(np.abs(u - u_exact)[2:-2, 2:-2])
        errors_du[i] = np.max(np.abs(dudx - dudx_exact)[2:-2, 2:-2]) + np.max(
            np.abs(dudy - dudy_exact)[2:-2, 2:-2]
        )

    print(f"Convergence (norm = {np.inf}):")
    print(f"{'N':>5} {'Err_u':>14} {'Order':>8} {'Err_du':>14} {'Order':>8}")
    print("-" * 55)
    for i, n in enumerate(n_range):
        if i == 0:
            order_u = np.nan
            order_du = np.nan
        else:
            order_u = np.log(errors_u[i - 1] / errors_u[i]) / np.log(2)
            order_du = np.log(errors_du[i - 1] / errors_du[i]) / np.log(2)

        print(
            f"{n:5d} "
            f"{errors_u[i]:14.2e} "
            f"{order_u:8.2f} "
            f"{errors_du[i]:14.2e} "
            f"{order_du:8.2f}"
        )

    plt.figure()
    plt.subplot(121)
    plt.plot(y[2:-2], u_exact[nx // 2, 2:-2], label="exact")
    plt.plot(y[2:-2], u[nx // 2, 2:-2], label="numerical", linestyle="--")
    plt.xlabel("y")
    plt.ylabel("u")
    plt.subplot(122)
    plt.plot(y[2:-2], dudy_exact[nx // 2, 2:-2], label="exact")
    plt.plot(y[2:-2], dudy[nx // 2, 2:-2], label="numerical", linestyle="--")
    plt.xlabel("y")
    plt.ylabel("du/dy")
    plt.legend()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.loglog(1 / n_range, errors_u, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $u$")
    plt.subplot(1, 2, 2)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $\\nabla u$")
    plt.show()


def convergence_test2():
    """
    exampel 1 in Cho.
    """
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact
    global a, b, a_tau, n1, n2
    global rows, cols, vals
    global surface, permittivity

    r = 0.5

    def surface(x, y):
        return x**2 + y**2 - r**2

    def permittivity(x, y):
        return 1.0

    n_range = 2 ** np.arange(4, 10, dtype=int)
    errors_u = np.zeros(n_range.size)
    errors_du = np.zeros(n_range.size)

    for i, n in enumerate(n_range):
        nx, ny = n, n
        dx, dy = 2.0 / nx, 2.0 / ny
        x = np.arange(-1.0 + dx / 2, 1.0 + dx / 2, dx)
        y = np.arange(-1.0 + dy / 2, 1.0 + dy / 2, dy)
        X, Y = np.meshgrid(x, y, indexing="ij")

        mask = X**2 + Y**2 < r**2
        f = np.zeros((nx, ny))
        u_exact = 1.0 + np.log(2 * np.sqrt(X**2 + Y**2))
        u_exact[mask] = 1.0
        dudx_exact = X / (X**2 + Y**2)
        dudx_exact[mask] = 0.0
        dudy_exact = Y / (X**2 + Y**2)
        dudy_exact[mask] = 0.0
        a = np.zeros((nx, ny))
        b = 2.0 * np.ones((nx, ny))
        n1, n2 = compute_normal_field()
        a_tau = compute_a_tau_field()

        rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
        A = construct_matrix()
        u = spsolve(A, f.flatten())
        u = u.reshape((nx, ny))
        dudx, dudy = gradient(u)
        # norm = 2
        norm = np.inf
        errors_u[i] = np.linalg.norm((u - u_exact)[2:-2, 2:-2].flat, norm)
        # errors_du[i] = np.linalg.norm(
        #     (dudx - dudx_exact)[2:-2, 2:-2].flat, norm
        # ) + np.linalg.norm((dudy - dudy_exact)[2:-2, 2:-2].flat, norm)
        errors_du[i] = np.linalg.norm(
            np.append(
                (dudx - dudx_exact)[2:-2, 2:-2].flat,
                (dudy - dudy_exact)[2:-2, 2:-2].flat,
            ),
            norm,
        )
        print(f"n={n}, Max error: {errors_u[i]}, Max grad error: {errors_du[i]}")

    print(f"Convergence (norm = {np.inf}):")
    print(f"{'N':>5} {'Err_u':>14} {'Order':>8} {'Err_du':>14} {'Order':>8}")
    print("-" * 55)
    for i, n in enumerate(n_range):
        if i == 0:
            order_u = np.nan
            order_du = np.nan
        else:
            order_u = np.log(errors_u[i - 1] / errors_u[i]) / np.log(2)
            order_du = np.log(errors_du[i - 1] / errors_du[i]) / np.log(2)

        print(
            f"{n:5d} "
            f"{errors_u[i]:14.2e} "
            f"{order_u:8.2f} "
            f"{errors_du[i]:14.2e} "
            f"{order_du:8.2f}"
        )

    plt.figure()
    plt.pcolormesh(X, Y, np.log(np.abs(dudx - dudx_exact)), shading="auto")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.title("Error in du/dx")

    plt.figure()
    plt.subplot(121)
    plt.loglog(1 / n_range, errors_u, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $u$")
    plt.subplot(122)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $\\nabla u$")

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    ax[0].plot_surface(X, Y, u_exact)
    ax[0].set_title("Exact solution")
    ax[1].plot_surface(X, Y, u)
    ax[1].set_title("Nmerical solution of 2D Poisson equation")

    plt.figure()
    plt.subplot(221)
    plt.pcolormesh(X, Y, dudx_exact, shading="auto")
    plt.colorbar()
    plt.title("Exact $du/dx$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(222)
    plt.pcolormesh(X, Y, dudx, shading="auto")
    plt.colorbar()
    plt.title("Nmerical $du/dx$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(223)
    plt.pcolormesh(X, Y, dudy_exact, shading="auto")
    plt.colorbar()
    plt.title("Exact $du/dy$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(224)
    plt.pcolormesh(X, Y, dudy, shading="auto")
    plt.colorbar()
    plt.title("Nmerical $du/dy$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def convergence_test3():
    """
    exampel 3 in Liu.
    """
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact
    global a, b, a_tau, n1, n2
    global rows, cols, vals
    global surface, permittivity

    r = 0.25

    def surface(x, y):
        return (x - 0.5) ** 2 + (y - 0.5) ** 2 - r**2

    def permittivity(x, y):
        return 2.0 if surface(x, y) <= 0 else 1.0

    n_range = 2 ** np.arange(4, 10, dtype=int)
    errors_u = np.zeros(n_range.size)
    errors_du = np.zeros(n_range.size)

    for i, n in enumerate(n_range):
        nx, ny = n, n
        dx, dy = 1.0 / nx, 1.0 / ny
        x = np.arange(dx / 2, 1.0 + dx / 2, dx)
        y = np.arange(dy / 2, 1.0 + dy / 2, dy)
        X, Y = np.meshgrid(x, y, indexing="ij")

        mask = (X - 0.5) ** 2 + (Y - 0.5) ** 2 > r**2
        f = 8 * (X**2 + Y**2 - 1.0) * np.exp(-(X**2 + Y**2))
        f[mask] = 0.0
        u_exact = np.exp(-(X**2 + Y**2))
        u_exact[mask] = 0.0
        dudx_exact = -2.0 * X * np.exp(-(X**2 + Y**2))
        dudx_exact[mask] = 0.0
        dudy_exact = -2.0 * Y * np.exp(-(X**2 + Y**2))
        dudy_exact[mask] = 0.0
        a = -np.exp(-(X**2 + Y**2))
        b = 8.0 * (2 * X**2 + 2 * Y**2 - X - Y) * np.exp(-(X**2 + Y**2))
        n1, n2 = compute_normal_field()
        a_tau = compute_a_tau_field()

        rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
        A = construct_matrix()
        u = spsolve(A, f.flatten())
        u = u.reshape((nx, ny))
        dudx, dudy = gradient(u)
        errors_u[i] = np.linalg.norm((u - u_exact)[2:-2, 2:-2].flat, np.inf)
        errors_du[i] = np.linalg.norm(
            np.append(
                (dudx - dudx_exact)[2:-2, 2:-2].flat,
                (dudy - dudy_exact)[2:-2, 2:-2].flat,
            ),
            np.inf,
        )
        print(f"n={n}, Max error: {errors_u[i]}, Max grad error: {errors_du[i]}")

    # convergence table
    print(f"Convergence (norm = {np.inf}):")
    print(f"{'N':>5} {'Err_u':>14} {'Order':>8} {'Err_du':>14} {'Order':>8}")
    print("-" * 55)

    for i, n in enumerate(n_range):
        if i == 0:
            order_u = np.nan
            order_du = np.nan
        else:
            order_u = np.log(errors_u[i - 1] / errors_u[i]) / np.log(2)
            order_du = np.log(errors_du[i - 1] / errors_du[i]) / np.log(2)

        print(
            f"{n:5d} "
            f"{errors_u[i]:14.2e} "
            f"{order_u:8.2f} "
            f"{errors_du[i]:14.2e} "
            f"{order_du:8.2f}"
        )

    plt.figure()
    plt.subplot(121)
    plt.loglog(1 / n_range, errors_u, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $u$")
    plt.subplot(122)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $\\nabla u$")

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    ax[0].plot_surface(X, Y, u_exact)
    ax[0].set_title("Exact solution")
    ax[1].plot_surface(X, Y, u)
    ax[1].set_title("Nmerical solution of 2D Poisson equation")
    plt.show()


def convergence_test4():
    """Example 4.2 in Cho et al. (2019): irregular interface Gamma.

    Domain: [-1, 1]^2.
    Interface: zero-level set of phi(x, y) = r - (0.5 + 0.15 sin(5*phi_polar)),
        with r = sqrt((x - 0.02*sqrt(5))^2 + (y - 0.02*sqrt(3))^2)
        and phi_polar the polar angle around (0.02*sqrt(5), 0.02*sqrt(3)).
    Solution:
        u^- = x^2 + y^2                                     if phi < 0
        u^+ = 0.1*(x^2 + y^2)^2 - 0.01*log(2*sqrt(x^2+y^2)) if phi >= 0
    Source f = 4 inside, 16(x^2+y^2) outside.
    Coefficients: beta^- = 1, beta^+ = 10.
    Jump conditions:
        [u]      = u^+ - u^-
        [beta u_n] = (4(x^2+y^2) - 0.1/(x^2+y^2) - 2)*(x*n_x + y*n_y)
    """
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact
    global a, b, a_tau, n1, n2
    global rows, cols, vals
    global surface, permittivity

    x0 = 0.02 * np.sqrt(5)
    y0 = 0.02 * np.sqrt(3)

    def surface(x, y):
        rr = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        ang = np.arctan2(y - y0, x - x0)
        return rr - (0.5 + 0.15 * np.sin(5 * ang))

    def permittivity(x, y):
        return 1.0 if surface(x, y) < 0 else 10.0

    n_range = 2 ** np.arange(4, 10, dtype=int)
    errors_u = np.zeros(n_range.size)
    errors_du = np.zeros(n_range.size)

    eps_safe = 1e-30
    for idx, n in enumerate(n_range[3:4]):
        nx, ny = n, n
        dx, dy = 2.0 / nx, 2.0 / ny
        x = np.arange(-1.0 + dx / 2, 1.0 + dx / 2, dx)
        y = np.arange(-1.0 + dy / 2, 1.0 + dy / 2, dy)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Region mask via the level-set sign at every grid point.
        Phi = surface(X, Y)
        mask_minus = Phi < 0  # Omega^-: inside the irregular shape.

        R2 = X**2 + Y**2
        R2_safe = np.maximum(R2, eps_safe)

        # u^+ - u^- and the two-sided u/grad fields, defined on the full grid.
        u_minus = R2.copy()
        u_plus = 0.1 * R2**2 - 0.01 * np.log(2.0 * np.sqrt(R2_safe))
        u_exact = np.where(mask_minus, u_minus, u_plus)

        dudx_minus = 2.0 * X
        dudy_minus = 2.0 * Y
        coeff_plus = 0.4 * R2 - 0.01 / R2_safe
        dudx_plus = X * coeff_plus
        dudy_plus = Y * coeff_plus
        dudx_exact = np.where(mask_minus, dudx_minus, dudx_plus)
        dudy_exact = np.where(mask_minus, dudy_minus, dudy_plus)

        # Right-hand side: f = div(beta grad u). Piecewise constant beta means
        # f = beta * Laplace(u) on each side.
        f = np.where(mask_minus, 4.0, 16.0 * R2)

        # Jump in u, evaluated as a smooth function over the whole grid (the
        # solver only samples it via cubic interpolation at interface points).
        a = u_plus - u_minus

        # Normals and a_tau need to be available before computing b, since the
        # paper formula for [beta u_n] involves the unit normal explicitly.
        n1, n2 = compute_normal_field()
        b_factor = 4.0 * R2 - 0.1 / R2_safe - 2.0
        b = (X * n1 + Y * n2) * b_factor
        a_tau = compute_a_tau_field()

        rows, cols, vals = [], [], []
        A = construct_matrix()
        u = spsolve(A, f.flatten())
        u = u.reshape((nx, ny))
        # dudx, dudy = gradient(u)
        dudx, dudy = np.zeros_like(u), np.zeros_like(u)

        errors_u[idx] = np.linalg.norm((u - u_exact)[2:-2, 2:-2].flat, np.inf)
        errors_du[idx] = np.linalg.norm(
            np.append(
                (dudx - dudx_exact)[2:-2, 2:-2].flat,
                (dudy - dudy_exact)[2:-2, 2:-2].flat,
            ),
            np.inf,
        )
        print(
            f"n={n}, Max error: {errors_u[idx]:.3e}, "
            f"Max grad error: {errors_du[idx]:.3e}"
        )

    print(f"Convergence (norm = {np.inf}):")
    print(f"{'N':>5} {'Err_u':>14} {'Order':>8} {'Err_du':>14} {'Order':>8}")
    print("-" * 55)
    for idx, n in enumerate(n_range):
        if idx == 0:
            order_u = np.nan
            order_du = np.nan
        else:
            order_u = np.log(errors_u[idx - 1] / errors_u[idx]) / np.log(2)
            order_du = np.log(errors_du[idx - 1] / errors_du[idx]) / np.log(2)
        print(
            f"{n:5d} "
            f"{errors_u[idx]:14.2e} "
            f"{order_u:8.2f} "
            f"{errors_du[idx]:14.2e} "
            f"{order_du:8.2f}"
        )

    plt.figure()
    plt.subplot(121)
    plt.loglog(1 / n_range, errors_u, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Example 4.2: convergence of $u$")
    plt.subplot(122)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Example 4.2: convergence of $\\nabla u$")

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    ax[0].plot_surface(X, Y, u_exact, edgecolor="black", cmap=cm.coolwarm)
    ax[0].set_title("Example 4.2: exact")
    ax[1].plot_surface(X, Y, u, edgecolor="black", cmap=cm.coolwarm)
    ax[1].set_title("Example 4.2: numerical")
    plt.show()


def convergence_test5():
    """Example 4.3 in Cho et al. (2019): general (non-constant) beta.

    Domain and interface are the same as Example 4.2.
    Coefficients:
        beta^- = exp(10*x)              if phi < 0
        beta^+ = sin(x + y) + 2         if phi >= 0
    Solution:
        u^- = sin(x + y)                if phi < 0
        u^+ = log(x^2 + y^2 + 1)        if phi >= 0
    Source f = div(beta grad u) computed analytically:
        f^- = -2*exp(10x)*sin(x+y) + 10*exp(10x)*cos(x+y)
        f^+ =  (sin(x+y)+2)*4/(x^2+y^2+1)^2
              + 2*cos(x+y)*(x+y)/(x^2+y^2+1)
    Jump conditions:
        [u]        = u^+ - u^-
        [beta u_n] = (sin(x+y)+2)*(2x n_x + 2y n_y)/(x^2+y^2+1)
                   - exp(10x)*cos(x+y)*(n_x + n_y)
    """
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact
    global a, b, a_tau, n1, n2
    global rows, cols, vals
    global surface, permittivity

    x0 = 0.02 * np.sqrt(5)
    y0 = 0.02 * np.sqrt(3)

    def surface(x, y):
        rr = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        ang = np.arctan2(y - y0, x - x0)
        return rr - (0.5 + 0.15 * np.sin(5 * ang))

    def permittivity(x, y):
        return np.exp(10.0 * x) if surface(x, y) < 0 else np.sin(x + y) + 2.0

    n_range = 2 ** np.arange(4, 10, dtype=int)
    errors_u = np.zeros(n_range.size)
    errors_du = np.zeros(n_range.size)

    for idx, n in enumerate(n_range):
        nx, ny = n, n
        dx, dy = 2.0 / nx, 2.0 / ny
        x = np.arange(-1.0 + dx / 2, 1.0 + dx / 2, dx)
        y = np.arange(-1.0 + dy / 2, 1.0 + dy / 2, dy)
        X, Y = np.meshgrid(x, y, indexing="ij")

        Phi = surface(X, Y)
        mask_minus = Phi < 0

        R2P1 = X**2 + Y**2 + 1.0  # x^2 + y^2 + 1 — strictly positive
        sxy = np.sin(X + Y)
        cxy = np.cos(X + Y)
        e10x = np.exp(10.0 * X)

        u_minus = sxy
        u_plus = np.log(R2P1)
        u_exact = np.where(mask_minus, u_minus, u_plus)

        dudx_minus = cxy
        dudy_minus = cxy
        dudx_plus = 2.0 * X / R2P1
        dudy_plus = 2.0 * Y / R2P1
        dudx_exact = np.where(mask_minus, dudx_minus, dudx_plus)
        dudy_exact = np.where(mask_minus, dudy_minus, dudy_plus)

        f_minus = -2.0 * e10x * sxy + 10.0 * e10x * cxy
        f_plus = (sxy + 2.0) * 4.0 / R2P1**2 + 2.0 * cxy * (X + Y) / R2P1
        f = np.where(mask_minus, f_minus, f_plus)

        a = u_plus - u_minus

        n1, n2 = compute_normal_field()
        b = (sxy + 2.0) * (2.0 * X * n1 + 2.0 * Y * n2) / R2P1 - e10x * cxy * (n1 + n2)
        a_tau = compute_a_tau_field()

        rows, cols, vals = [], [], []
        A = construct_matrix()
        u = spsolve(A, f.flatten())
        u = u.reshape((nx, ny))
        dudx, dudy = gradient(u)

        errors_u[idx] = np.linalg.norm((u - u_exact)[2:-2, 2:-2].flat, np.inf)
        errors_du[idx] = np.linalg.norm(
            np.append(
                (dudx - dudx_exact)[2:-2, 2:-2].flat,
                (dudy - dudy_exact)[2:-2, 2:-2].flat,
            ),
            np.inf,
        )
        print(
            f"n={n}, Max error: {errors_u[idx]:.3e}, "
            f"Max grad error: {errors_du[idx]:.3e}"
        )

    print(f"Convergence (norm = {np.inf}):")
    print(f"{'N':>5} {'Err_u':>14} {'Order':>8} {'Err_du':>14} {'Order':>8}")
    print("-" * 55)
    for idx, n in enumerate(n_range):
        if idx == 0:
            order_u = np.nan
            order_du = np.nan
        else:
            order_u = np.log(errors_u[idx - 1] / errors_u[idx]) / np.log(2)
            order_du = np.log(errors_du[idx - 1] / errors_du[idx]) / np.log(2)
        print(
            f"{n:5d} "
            f"{errors_u[idx]:14.2e} "
            f"{order_u:8.2f} "
            f"{errors_du[idx]:14.2e} "
            f"{order_du:8.2f}"
        )

    plt.figure()
    plt.subplot(121)
    plt.loglog(1 / n_range, errors_u, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Example 4.3: convergence of $u$")
    plt.subplot(122)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Example 4.3: convergence of $\\nabla u$")

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    ax[0].plot_surface(X, Y, u_exact, edgecolor="black", cmap=cm.coolwarm)
    ax[0].set_title("Example 4.3: exact")
    ax[1].plot_surface(X, Y, u, edgecolor="black", cmap=cm.coolwarm)
    ax[1].set_title("Example 4.3: numerical")
    plt.show()

def test_case4(n: int=64):
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact
    global a, b, a_tau, n1, n2
    global rows, cols, vals
    global surface, permittivity

    x0 = 0.02 * np.sqrt(5)
    y0 = 0.02 * np.sqrt(3)

    def surface(x, y):
        rr = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        ang = np.arctan2(y - y0, x - x0)
        return rr - (0.5 + 0.15 * np.sin(5 * ang))

    def permittivity(x, y):
        return 1.0 if surface(x, y) < 0 else 10.0

    G = 3
    eps_safe = 1e-30
    n = n + 6
    nx, ny = n, n
    dx, dy = 2.0 / (nx - 2*G), 2.0 / (ny - 2*G)
    x = np.arange(-1.0 + dx / 2 -G*dx, 1.0 + dx / 2 + G*dx, dx)
    y = np.arange(-1.0 + dy / 2 -G*dy, 1.0 + dy / 2 + G*dy, dy)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Region mask via the level-set sign at every grid point.
    Phi = surface(X, Y)
    mask_minus = Phi < 0  # Omega^-: inside the irregular shape.

    R2 = X**2 + Y**2
    R2_safe = np.maximum(R2, eps_safe)

    # u^+ - u^- and the two-sided u/grad fields, defined on the full grid.
    u_minus = R2.copy()
    u_plus = 0.1 * R2**2 - 0.01 * np.log(2.0 * np.sqrt(R2_safe))
    u_exact = np.where(mask_minus, u_minus, u_plus)
    dudx_minus = 2.0 * X
    dudy_minus = 2.0 * Y
    coeff_plus = 0.4 * R2 - 0.01 / R2_safe
    dudx_plus = X * coeff_plus
    dudy_plus = Y * coeff_plus
    dudx_exact = np.where(mask_minus, dudx_minus, dudx_plus)
    dudy_exact = np.where(mask_minus, dudy_minus, dudy_plus)


    # Right-hand side: f = div(beta grad u). Piecewise constant beta means
    # f = beta * Laplace(u) on each side.
    f = np.where(mask_minus, 4.0, 16.0 * R2)

    # Jump in u, evaluated as a smooth function over the whole grid (the
    # solver only samples it via cubic interpolation at interface points).
    a = u_plus - u_minus

    # Normals and a_tau need to be available before computing b, since the
    # paper formula for [beta u_n] involves the unit normal explicitly.
    n1, n2 = compute_normal_field()
    b_factor = 4.0 * R2 - 0.1 / R2_safe - 2.0
    b = (X * n1 + Y * n2) * b_factor
    a_tau = compute_a_tau_field()

    rows, cols, vals = [], [], []
    A = construct_matrix()
    # # DEBUG: print RHS for comparison with C++
    # print(f"=== RHS VECTOR (nx={nx}, ny={ny}) ===")
    # for ii in range(nx):
    #     for jj in range(ny):
    #         print(f"rhs[{ii},{jj}]={float(f[ii, jj]):.15e}")
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    dudx, dudy = gradient(u)
    
    error_u = np.linalg.norm((u - u_exact)[G:-G, G:-G].flat, np.inf)
    error_du = np.linalg.norm(
        np.append(
            (dudx - dudx_exact)[G:-G, G:-G].flat,
            (dudy - dudy_exact)[G:-G, G:-G].flat,
        ),
        np.inf,
    )
    print(
        f"Max error: {error_u:.3e}, "
        f"Max grad error: {error_du:.3e}"
    )

    # fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    # ax[0].plot_surface(X, Y, u_exact, edgecolor="black", cmap=cm.coolwarm)
    # ax[0].set_title("Example 4.2: exact")
    # ax[1].plot_surface(X, Y, u, edgecolor="black", cmap=cm.coolwarm)
    # ax[1].set_title("Example 4.2: numerical")
    # plt.show()






if __name__ == "__main__":
    # convergence_test1()
    # convergence_test2()
    # convergence_test3()
    # convergence_test4()
    # convergence_test5()
    import sys
    if len(sys.argv) > 1:
        test_case4(int(sys.argv[1]))
    else:
        test_case4()

import enum

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

EPS = 0.01


class Direction(enum.IntFlag):
    R = 1 << 0  # 0001
    T = 1 << 1  # 0010
    L = 1 << 2  # 0100
    B = 1 << 3  # 1000


def dirsign(direction: int):
    if direction == Direction.R or direction == Direction.T:
        return 1.0
    elif direction == Direction.L or direction == direction.B:
        return -1.0
    else:
        raise ValueError("Invalid direction for dirsign", direction)


def index(i: int, j: int) -> int:
    """flatten index"""
    return i * ny + j


def center(i: int, j: int) -> tuple[float, float]:
    return x[i], y[j]


def surface(x: float, y: float) -> float:
    return 1.0


def normal(x: float, y: float) -> tuple[float, float]:
    return x, y


def permittivity(x: float, y: float) -> float:
    return 1.0


def compute_a_tau_field() -> np.ndarray:
    """Compute tangential derivative of jump condition a at (i, j)"""
    a_tau = np.zeros((nx, ny))
    for i in range(2, nx - 2):
        for j in range(2, ny - 2):
            dx_a = (-a[i + 2, j] + 8 * a[i + 1, j] - 8 * a[i - 1, j] + a[i - 2, j]) / (
                12 * dx
            )
            dy_a = (-a[i, j + 2] + 8 * a[i, j + 1] - 8 * a[i, j - 1] + a[i, j - 2]) / (
                12 * dy
            )
            a_tau[i, j] = -dx_a * n2[i, j] + dy_a * n1[i, j]
    return a_tau


def compute_normal_field() -> tuple[np.ndarray, np.ndarray]:
    n1 = np.zeros((nx, ny))
    n2 = np.zeros((nx, ny))
    for i in range(2, nx - 2):
        for j in range(2, ny - 2):
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


def compute_theta(direction: int, i: int, j: int) -> float:
    x, y = center(i, j)
    eta = surface(x, y)
    if direction == Direction.R or direction == Direction.L:
        eta_r = surface(x + dx, y)
        eta_l = surface(x - dx, y)
        d_eta = (eta_r - eta_l) / 2
        dd_eta = (eta_r - 2 * eta + eta_l) / 2
    elif direction == Direction.T or direction == Direction.B:
        eta_t = surface(x, y + dy)
        eta_b = surface(x, y - dy)
        d_eta = (eta_t - eta_b) / 2
        dd_eta = (eta_t - 2 * eta + eta_b) / 2
    else:
        raise ValueError("Invalid direction for compute_theta", direction)

    if np.isclose(dd_eta, 0.0):
        theta = np.abs(eta / d_eta)
    else:
        theta = (
            -dirsign(direction) * d_eta
            - np.sign(eta) * np.sqrt(d_eta**2 - 4 * dd_eta * eta)
        ) / (2 * dd_eta)
    if theta < 1e-6 or theta > 1.0 - 1e-6:
        theta = 1.0
        # breakpoint()
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
    if direction == Direction.R or direction == Direction.L:
        k = i
        points = (
            field[k - 1 : k + 3, j]
            if dirsign(direction) > 0
            else field[k - 2 : k + 2, j][::-1]
        )
    else:
        k = j
        points = (
            field[i, k - 1 : k + 3]
            if dirsign(direction) > 0
            else field[i, k - 2 : k + 2][::-1]
        )
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


def construct_matrix():
    for i in range(nx):
        for j in range(ny):
            if i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2:
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
                    # extra = case3_extra_dir(direction, i, j)
                    # if extra is None:
                    #     coeff_case2(direction, i, j)
                    # else:
                    #     coeff_case3(direction, extra, i, j)
                    coeff_case2(direction, i, j)
                # case 3:
                #     coeff_case4(direction, i, j)
                case _:
                    raise NotImplementedError(
                        "All four sides cut at one cell, use finer grid"
                    )

    A = coo_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    # convert csr for cg / gmres solve
    # convert to csc for lu solve
    return A.tocsr()


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


def coeff_case1(direction: int, i: int, j: int) -> None:
    x, y = center(i, j)
    row_idx = index(i, j)
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
    B = (
        np.sign(eta)
        * s
        * (
            eps_p * (3 - 2 * theta) / ((1 - theta) * (2 - theta))
            + eps_m * (2 * theta + 1) / (theta * (theta + 1))
        )
    )
    C = (
        -np.sign(eta)
        * s
        * np.array(
            [
                -eps_m * theta / (1 + theta),
                eps_m * (1 + theta) / theta,
                eps_p * (2 - theta) / (1 - theta),
                -eps_p * (1 - theta) / (2 - theta),
            ]
        )
    )
    a_term = -s * a_I * eps_p * (3 - 2 * theta) / ((1 - theta) * (2 - theta))

    # geometric [eps*u_x] and [eps*u_y] Eq(34)
    P_inv = compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext)
    grad_tau = np.array(
        [-2 * x * n2_I, x * n1_I - y * n2_I, 2 * y * n1_I, -n2_I, n1_I, 0.0]
    )

    if is_x_dir:
        grad_coeff = -dx * eps_jump * n2_I * grad_tau @ P_inv
        a_tau_term = -dx * a_tau_I * eps_p * n2_I
        b_term = dx * b_I * n1_I
    else:
        grad_coeff = dy * eps_jump * n1_I * grad_tau @ P_inv
        a_tau_term = dy * a_tau_I * eps_p * n1_I
        b_term = dy * b_I * n2_I

    d = a_tau_term + b_term - a_term

    _grad_idx = {Direction.R: 0, Direction.L: 1, Direction.T: 2, Direction.B: 3}
    grad_coeff_dir = grad_coeff[_grad_idx[direction]]
    M = B - grad_coeff_dir

    offset = lambda ox, oy: (ox + 2) * 5 + (oy + 2)
    N = np.zeros(25)
    N[offset(1, 0)] = grad_coeff[0] if direction != Direction.R else 0.0
    N[offset(-1, 0)] = grad_coeff[1] if direction != Direction.L else 0.0
    N[offset(0, 1)] = grad_coeff[2] if direction != Direction.T else 0.0
    N[offset(0, -1)] = grad_coeff[3] if direction != Direction.B else 0.0
    N[offset(0, 0)] = grad_coeff[4]
    N[offset(*offset_ext)] = grad_coeff[5]

    # Place C[k] along the direction ray at offsets (-1, 0, 1, 2) from center
    _dir_step = {
        Direction.R: (1, 0),
        Direction.T: (0, 1),
        Direction.L: (-1, 0),
        Direction.B: (0, -1),
    }
    dx_dir, dy_dir = _dir_step[direction]
    for k in range(4):
        N[offset((k - 1) * dx_dir, (k - 1) * dy_dir)] -= C[k]

    M_inv_N = N / M
    M_inv_d = d / M

    # Substitution coefficient: eps in current direction / theta / denominator
    _eps_dir = {
        Direction.R: eps_r,
        Direction.L: eps_l,
        Direction.T: eps_t,
        Direction.B: eps_b,
    }
    _theta_dir = {
        Direction.R: theta_r,
        Direction.L: theta_l,
        Direction.T: theta_t,
        Direction.B: theta_b,
    }
    sub_coeff = (
        _eps_dir[direction] / _theta_dir[direction] / (bot_x if is_x_dir else bot_y)
    )
    f[i, j] -= M_inv_d * sub_coeff

    # Extra stencil entries for directions orthogonal to the current one
    add_terms = {}
    if direction != Direction.R:
        add_terms[(1, 0)] = eps_r / theta_r / bot_x
    if direction != Direction.L:
        add_terms[(-1, 0)] = eps_l / theta_l / bot_x
    if direction != Direction.T:
        add_terms[(0, 1)] = eps_t / theta_t / bot_y
    if direction != Direction.B:
        add_terms[(0, -1)] = eps_b / theta_b / bot_y

    for offset_x in range(-2, 3):
        for offset_y in range(-2, 3):
            value = M_inv_N[offset(offset_x, offset_y)] * sub_coeff
            if (offset_x, offset_y) == (0, 0):
                value += (
                    -(eps_r / theta_r + eps_l / theta_l) / bot_x
                    - (eps_t / theta_t + eps_b / theta_b) / bot_y
                )
            else:
                value += add_terms.get((offset_x, offset_y), 0.0)
            rows.append(row_idx)
            cols.append(index(i + offset_x, j + offset_y))
            vals.append(value)


def coeff_case2(direction: int, i: int, j: int) -> None:
    """coeff of u_ij and its neighbors for a case 2 cell"""
    x, y = center(i, j)
    eta = surface(x, y)
    row_idx = index(i, j)  # laplacian matrix row index

    D = np.zeros(2)
    M = np.zeros((2, 2))
    N = np.zeros((2, 25))

    # used to traverse N matrix
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
    eps_jump = [eps_p[d] - eps_m[d] for d in range(2)]

    # normal evaluated at x_R and x_y
    n1_I = np.array([interp(dir[d], theta[d], i, j, n1) for d in range(2)])
    n2_I = np.array([interp(dir[d], theta[d], i, j, n2) for d in range(2)])

    # a_yau at x and y
    a_tau_I = [interp(dir[d], theta[d], i, j, a_tau) for d in range(2)]

    # jump conditions at x_x and x_y
    a_I = [interp(dir[d], theta[d], i, j, a) for d in range(2)]
    b_I = [interp(dir[d], theta[d], i, j, b) for d in range(2)]

    # algebraic [eps*u_x] and [eps*u_y] Eq(30)
    B = [
        (
            s_eta
            * s[d]
            * (
                eps_p[d] * (3 - 2 * theta[d]) / ((1 - theta[d]) * (2 - theta[d]))
                + eps_m[d] * (2 * theta[d] + 1) / (theta[d] * (theta[d] + 1))
            )
        )
        for d in range(2)
    ]
    C = [
        (
            -s_eta
            * s[d]
            * np.array(
                [
                    -eps_m[d] * theta[d] / (1 + theta[d]),
                    eps_m[d] * (1 + theta[d]) / theta[d],
                    eps_p[d] * (2 - theta[d]) / (1 - theta[d]),
                    -eps_p[d] * (1 - theta[d]) / (2 - theta[d]),
                ]
            )
        )
        for d in range(2)
    ]
    a_term = [
        -s[d] * a_I[d] * eps_p[d] * (3 - 2 * theta[d]) / ((1 - theta[d]) * (2 - theta[d]))
        for d in range(2)
    ]

    P_inv = compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext)
    grad_tau = [
        np.array(
            [
                -2 * x * n2_I[d],
                x * n1_I[d] - y * n2_I[d],
                2 * y * n1_I[d],
                -n2_I[d],
                n1_I[d],
                0.0,
            ]
        )
        for d in range(2)
    ]
    n_norm = [n1_I[0], n2_I[1]]
    n_tang = [-n2_I[0], n1_I[1]]
    grad_coeff = [dr[d] * eps_jump[d] * n_tang[d] * grad_tau[d] @ P_inv for d in range(2) ]
    a_tau_term = [dr[d] * a_tau_I[d] * eps_p[d] * n_tang[d] for d in range(2)]
    b_term = [dr[d] * b_I[d] * n_norm[d] for d in range(2)]

    D[:] = [ a_tau_term[d] + b_term[d] - a_term[d] for d in range(2) ]

    _grad_idx = {Direction.R: 0, Direction.L: 1, Direction.T: 2, Direction.B: 3}
    M[0, 0] = B[0] - grad_coeff[0][_grad_idx[dir[0]]]
    M[1, 1] = B[1] - grad_coeff[1][_grad_idx[dir[1]]]
    M[0, 1] = -grad_coeff[0][_grad_idx[dir[1]]]
    M[1, 0] = -grad_coeff[1][_grad_idx[dir[0]]]

    _dir_step = {
        Direction.R: (1, 0),
        Direction.T: (0, 1),
        Direction.L: (-1, 0),
        Direction.B: (0, -1),
    }
    for d in range(2):
        N[d, offset(1, 0)] = grad_coeff[d][0] if dir[0] != Direction.R else 0.0
        N[d, offset(-1, 0)] = grad_coeff[d][1] if dir[0] != Direction.L else 0.0
        N[d, offset(0, 1)] = grad_coeff[d][2] if dir[1] != Direction.T else 0.0
        N[d, offset(0, -1)] = grad_coeff[d][3] if dir[1] != Direction.B else 0.0
        N[d, offset(0, 0)] = grad_coeff[d][4]
        N[d, offset(*offset_ext)] = grad_coeff[d][5]

        # Place [k] along the direction ray at offsets (-1, 0, 1, 2) from center
        dx_dir, dy_dir = _dir_step[dir[d]]
        for k in range(4):
            N[d, offset((k - 1) * dx_dir, (k - 1) * dy_dir)] -= C[d][k]

    M_inv_d = np.linalg.solve(M, D)
    M_inv_N = np.linalg.solve(M, N)

    sub_coeff = [ eps[0] / theta[0] / bot_x, eps[1] / theta[1] / bot_y ]
    f[i, j] -= M_inv_d[0] * sub_coeff[0] + M_inv_d[1] * sub_coeff[1]

    # Extra stencil entries for directions orthogonal to the current one
    add_terms = {}
    if dir[0] != Direction.R:
        add_terms[(1, 0)] = eps_r / theta_r / bot_x
    if dir[0] != Direction.L:
        add_terms[(-1, 0)] = eps_l / theta_l / bot_x
    if dir[1] != Direction.T:
        add_terms[(0, 1)] = eps_t / theta_t / bot_y
    if dir[1] != Direction.B:
        add_terms[(0, -1)] = eps_b / theta_b / bot_y

    for offset_x in range(-2, 3):
        for offset_y in range(-2, 3):
            value = (
                    M_inv_N[0, offset(offset_x, offset_y)] * sub_coeff[0]
                    + M_inv_N[1, offset(offset_x, offset_y)] * sub_coeff[1]
                )
            if (offset_x, offset_y) == (0, 0):
                value += (
                    -(eps_r / theta_r + eps_l / theta_l) / bot_x
                    - (eps_t / theta_t + eps_b / theta_b) / bot_y
                )
            else:
                value += add_terms.get((offset_x, offset_y), 0.0)
            rows.append(row_idx)
            cols.append(index(i + offset_x, j + offset_y))
            vals.append(value)

def coeff_case3(direction: int, extra_dir: int, i: int, j: int) -> None:
    pass


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
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
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
                    # extra = case3_extra_dir(direction, i, j)
                    # if extra is None:
                    #     u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                    #         interface_value_case2(direction, i, j, u)
                    #     )
                    # else:
                    #     u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                    #         interface_value_case3(direction, extra, i, j, u)
                    #     )
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                        interface_value_case2(direction, i, j, u)
                    )
                # case 3:
                #     u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                #         interface_value_case4(direction, i, j, u)
                #     )
                case _:
                    raise NotImplementedError(
                        "All four sides cut at one cell — beyond case 4."
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


def interface_value_case0(
    i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    return u[i - 1, j], u[i + 1, j], u[i, j - 1], u[i, j + 1], 1.0, 1.0, 1.0, 1.0


def interface_value_case1(
    direction: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    x, y = center(i, j)
    eta = surface(x, y)  # assume this is negative for now
    theta = compute_theta(direction, i, j)
    a_tau_I = interp(direction, theta, i, j, a_tau)
    a_I = interp(direction, theta, i, j, a)
    b_I = interp(direction, theta, i, j, b)
    n1_I = interp(direction, theta, i, j, n1)
    n2_I = interp(direction, theta, i, j, n2)

    if direction == Direction.R:
        theta_l, theta_r, theta_b, theta_t = 1.0, theta, 1.0, 1.0

        if eta > 0:
            _eps_p = permittivity(x, y)
            _eps_m = permittivity(x + dx, y)
            eps_jump = _eps_p - _eps_m
            # swap these two variable in the d expression
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x + dx, y)
            _eps_m = permittivity(x, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            -a_tau_I * eps_p * n2_I * dx
            + b_I * n1_I * dx
            + a_I * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r))
        )

        if eta > 0:
            # in the following formulas, permittivity signs are also swapped
            # the permittivity jump stays the same
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            -eps_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r))
            - eps_m * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
            - eps_jump * n2_I**2 * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
        )

        N = [
            # u[i,j]
            -eps_jump * n1_I * n2_I * theta_r * dx / dy
            - (eps_jump * n2_I**2 + eps_m) * (1 + theta_r) / theta_r,
            # u[i+1,j]
            -eps_p * (theta_r - 2) / (theta_r - 1),
            # u[i+2,j]
            eps_p * (theta_r - 1) / (theta_r - 2),
            # u[i-1,j]
            eps_jump * n1_I * n2_I * theta_r * dx / dy
            + (eps_jump * n2_I**2 + eps_m) * theta_r / (1 + theta_r),
            # u[i,j-1]
            eps_jump * n1_I * n2_I * (2 * theta_r + 1) * dx / (2 * dy),
            # u[i,j+1]
            -eps_jump * n1_I * n2_I * dx / (2 * dy),
            # u[i-1,j-1]
            -eps_jump * n1_I * n2_I * theta_r * dx / dy,
        ]
        u_arr = [
            u[i, j],
            u[i + 1, j],
            u[i + 2, j],
            u[i - 1, j],
            u[i, j - 1],
            u[i, j + 1],
            u[i - 1, j - 1],
        ]
        u_I = (np.dot(N, u_arr) + d) / M
        return (
            u[i - 1, j],
            u_I,
            u[i, j - 1],
            u[i, j + 1],
            theta_l,
            theta_r,
            theta_b,
            theta_t,
        )
    elif direction == Direction.T:
        theta_l, theta_r, theta_b, theta_t = 1.0, 1.0, 1.0, theta

        if eta > 0:
            _eps_p = permittivity(x, y)
            _eps_m = permittivity(x, y + dy)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x, y + dy)
            _eps_m = permittivity(x, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            a_tau_I * eps_p * n1_I * dy
            + b_I * n2_I * dy
            + a_I * eps_p * (3 - 2 * theta_t) / ((2 - theta_t) * (1 - theta_t))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            -eps_p * (3 - 2 * theta_t) / ((1 - theta_t) * (2 - theta_t))
            - eps_m * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
            - eps_jump * n1_I**2 * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
        )
        N = [
            # u[i,j]
            -eps_jump * n1_I * n2_I * theta_t * dy / dx
            - (eps_jump * n1_I**2 + eps_m) * (1 + theta_t) / theta_t,
            # u[i,j+1]
            -eps_p * (theta_t - 2) / (theta_t - 1),
            # u[i,j+2]
            eps_p * (theta_t - 1) / (theta_t - 2),
            # u[i,j-1]
            eps_jump * n1_I * n2_I * theta_t * dy / dx
            + (eps_jump * n1_I**2 + eps_m) * theta_t / (1 + theta_t),
            # u[i-1,j]
            eps_jump * n1_I * n2_I * (2 * theta_t + 1) * dy / (2 * dx),
            # u[i+1,j]
            -eps_jump * n1_I * n2_I * dy / (2 * dx),
            # u[i-1,j-1]
            -eps_jump * n1_I * n2_I * theta_t * dy / dx,
        ]
        u_arr = [
            u[i, j],
            u[i, j + 1],
            u[i, j + 2],
            u[i, j - 1],
            u[i - 1, j],
            u[i + 1, j],
            u[i - 1, j - 1],
        ]
        u_I = (np.dot(N, u_arr) + d) / M
        return (
            u[i - 1, j],
            u[i + 1, j],
            u[i, j - 1],
            u_I,
            theta_l,
            theta_r,
            theta_b,
            theta_t,
        )
    elif direction == Direction.L:
        theta_l, theta_r, theta_b, theta_t = theta, 1.0, 1.0, 1.0

        if eta > 0:
            _eps_p = permittivity(x, y)
            _eps_m = permittivity(x - dx, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x - dx, y)
            _eps_m = permittivity(x, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            -a_tau_I * eps_p * n2_I * dx
            + b_I * n1_I * dx
            - a_I * eps_p * (3 - 2 * theta_l) / ((2 - theta_l) * (1 - theta_l))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            eps_p * (3 - 2 * theta_l) / ((1 - theta_l) * (2 - theta_l))
            + eps_m * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
            + eps_jump * n2_I**2 * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
        )

        N = [
            # u[i,j]
            -eps_jump * n1_I * n2_I * theta_l * dx / dy
            + (eps_jump * n2_I**2 + eps_m) * (1 + theta_l) / theta_l,
            # u[i-1,j]
            eps_p * (theta_l - 2) / (theta_l - 1),
            # u[i-2,j]
            -eps_p * (theta_l - 1) / (theta_l - 2),
            # u[i+1,j]
            eps_jump * n1_I * n2_I * theta_l * dx / dy
            - (eps_jump * n2_I**2 + eps_m) * theta_l / (1 + theta_l),
            # u[i,j-1]
            eps_jump * n1_I * n2_I * (2 * theta_l + 1) * dx / (2 * dy),
            # u[i,j+1]
            -eps_jump * n1_I * n2_I * dx / (2 * dy),
            # u[i+1,j-1]
            -eps_jump * n1_I * n2_I * theta_l * dx / dy,
        ]
        u_arr = [
            u[i, j],
            u[i - 1, j],
            u[i - 2, j],
            u[i + 1, j],
            u[i, j - 1],
            u[i, j + 1],
            u[i + 1, j - 1],
        ]
        u_I = (np.dot(N, u_arr) + d) / M
        return (
            u_I,
            u[i + 1, j],
            u[i, j - 1],
            u[i, j + 1],
            theta_l,
            theta_r,
            theta_b,
            theta_t,
        )
    elif direction == Direction.B:
        theta_l, theta_r, theta_b, theta_t = 1.0, 1.0, theta, 1.0

        if eta > 0:
            _eps_p = permittivity(x, y)
            _eps_m = permittivity(x, y - dy)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x, y - dy)
            _eps_m = permittivity(x, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            a_tau_I * eps_p * n1_I * dy
            + b_I * n2_I * dy
            - a_I * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            eps_p * (3 - 2 * theta_b) / ((1 - theta_b) * (2 - theta_b))
            + eps_m * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
            + eps_jump * n1_I**2 * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
        )
        N = [
            # u[i,j]
            -eps_jump * n1_I * n2_I * theta_b * dy / dx
            + (eps_jump * n1_I**2 + eps_m) * (1 + theta_b) / theta_b,
            # u[i,j-1]
            eps_p * (theta_b - 2) / (theta_b - 1),
            # u[i,j-2]
            -eps_p * (theta_b - 1) / (theta_b - 2),
            # u[i,j+1]
            eps_jump * n1_I * n2_I * theta_b * dy / dx
            - (eps_jump * n1_I**2 + eps_m) * theta_b / (1 + theta_b),
            # u[i-1,j]
            eps_jump * n1_I * n2_I * (2 * theta_b + 1) * dy / (2 * dx),
            # u[i+1,j]
            -eps_jump * n1_I * n2_I * dy / (2 * dx),
            # u[i-1,j+1]
            -eps_jump * n1_I * n2_I * theta_b * dy / dx,
        ]
        u_arr = [
            u[i, j],
            u[i, j - 1],
            u[i, j - 2],
            u[i, j + 1],
            u[i - 1, j],
            u[i + 1, j],
            u[i - 1, j + 1],
        ]
        u_I = (np.dot(N, u_arr) + d) / M
        return (
            u[i - 1, j],
            u[i + 1, j],
            u_I,
            u[i, j + 1],
            theta_l,
            theta_r,
            theta_b,
            theta_t,
        )
    else:
        raise ValueError("Invalid direction for case 1", direction)

def interface_value_case2(
    direction: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    x, y = center(i, j)
    eta = surface(x, y)

    d = np.zeros(2)
    M = np.zeros((2, 2))
    N = np.zeros((2, 25))

    # used to traverse N matrix
    offset = lambda offset_x, offset_y: (offset_x + 2) * 5 + (offset_y + 2)

    if direction == Direction.R | Direction.T:
        theta_r = compute_theta(Direction.R, i, j)
        theta_t = compute_theta(Direction.T, i, j)
        theta_l = 1.0
        theta_b = 1.0

        # normal evaluated at x_R and x_T
        n1_x = interp(Direction.R, theta_r, i, j, n1)
        n2_x = interp(Direction.R, theta_r, i, j, n2)
        n1_y = interp(Direction.T, theta_t, i, j, n1)
        n2_y = interp(Direction.T, theta_t, i, j, n2)

        # a_tau at x_R and x_T
        a_tau_x = interp(Direction.R, theta_r, i, j, a_tau)
        a_tau_y = interp(Direction.T, theta_t, i, j, a_tau)

        # jump conditions at x_R and x_T
        a_x = interp(Direction.R, theta_r, i, j, a)
        a_y = interp(Direction.T, theta_t, i, j, a)
        b_x = interp(Direction.R, theta_r, i, j, b)
        b_y = interp(Direction.T, theta_t, i, j, b)

        if eta > 0:
            _eps_p = permittivity(x, y)
            _eps_m = permittivity(x + dx, y + dy)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x + dx, y + dy)
            _eps_m = permittivity(x, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d[0] = (
            -a_tau_x * eps_p * n2_x * dx
            + b_x * n1_x * dx
            + a_x * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r))
        )
        d[1] = (
            a_tau_y * eps_p * n1_y * dy
            + b_y * n2_y * dy
            + a_y * eps_p * (3 - 2 * theta_t) / ((2 - theta_t) * (1 - theta_t))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M[0, 0] = (
            -eps_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r))
            - eps_m * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
            - eps_jump * n2_x**2 * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
        )
        M[0, 1] = eps_jump * n1_x * n2_x * dx / (dy * theta_t * (theta_t + 1))
        M[1, 0] = eps_jump * n1_y * n2_y * dy / (dx * theta_r * (theta_r + 1))
        M[1, 1] = (
            -eps_p * (3 - 2 * theta_t) / ((1 - theta_t) * (2 - theta_t))
            - eps_m * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
            - eps_jump * n1_y**2 * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
        )

        # fmt: off
        # u[i,j]
        N[0, offset(0, 0)] = -(eps_m + eps_jump * n2_x**2) * (theta_r + 1) / theta_r \
            - (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_t * theta_r + theta_t - 1) / theta_t)
        # u[i+1,j]
        N[0, offset(1, 0)] = -eps_p * (theta_r - 2) / (theta_r - 1)
        # u[i+2,j]
        N[0, offset(2, 0)] = eps_p * (theta_r - 1) / (theta_r - 2)
        # u[i-1,j]
        N[0, offset(-1, 0)] = (eps_m + eps_jump * n2_x**2) * theta_r / (theta_r + 1) \
            + eps_jump * n1_x * n2_x * theta_r * (dx / dy)
        # u[i,j-1]
        N[0, offset(0, -1)] = eps_jump * n1_x * n2_x * (dx / dy) * (theta_t / (theta_t + 1) + theta_r) 
        # u[i-1,j-1]
        N[0, offset(-1, -1)] = -eps_jump * n1_x * n2_x * theta_r * (dx / dy)

        # u[i,j]
        N[1, offset(0, 0)] = -(eps_m + eps_jump * n1_y**2) * (theta_t + 1) / theta_t \
            - (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_r * theta_t + theta_r - 1) / theta_r)
        # u[i,j+1]
        N[1, offset(0, 1)] = -eps_p * (theta_t - 2) / (theta_t - 1)
        # u[i,j+2]
        N[1, offset(0, 2)] = eps_p * (theta_t - 1) / (theta_t - 2)
        # u[i,j-1]
        N[1, offset(0, -1)] = (eps_m + eps_jump * n1_y**2) * theta_t / (theta_t + 1) \
            + eps_jump * n1_y * n2_y * theta_t * (dy / dx)
        # u[i-1,j]
        N[1, offset(-1, 0)] = eps_jump * n1_y * n2_y * (dy / dx) * (theta_r / (theta_r + 1) + theta_t)
        # u[i-1,j-1]
        N[1, offset(-1, -1)] = -eps_jump * n1_y * n2_y * theta_t * (dy / dx)
        # fmt: on

        u_arr = np.array(
            [
                u[i + offset_x, j + offset_y]
                for offset_x in range(-2, 3)
                for offset_y in range(-2, 3)
            ]
        )
        u_I = np.linalg.solve(M, N @ u_arr + d)
        return (
            u[i - 1, j],
            u_I[0],
            u[i, j - 1],
            u_I[1],
            theta_l,
            theta_r,
            theta_b,
            theta_t,
        )

    elif direction == Direction.L | Direction.T:
        theta_l = compute_theta(Direction.L, i, j)
        theta_t = compute_theta(Direction.T, i, j)
        theta_r = 1.0
        theta_b = 1.0

        # normal evaluated at x_L and x_T
        n1_x = interp(Direction.L, theta_l, i, j, n1)
        n2_x = interp(Direction.L, theta_l, i, j, n2)
        n1_y = interp(Direction.T, theta_t, i, j, n1)
        n2_y = interp(Direction.T, theta_t, i, j, n2)

        # jump conditions at x_L and x_T
        a_x = interp(Direction.L, theta_l, i, j, a)
        a_y = interp(Direction.T, theta_t, i, j, a)
        b_x = interp(Direction.L, theta_l, i, j, b)
        b_y = interp(Direction.T, theta_t, i, j, b)

        # a_tau at x_L and x_T
        a_tau_x = interp(Direction.L, theta_l, i, j, a_tau)
        a_tau_y = interp(Direction.T, theta_t, i, j, a_tau)

        if eta > 0:
            _eps_p = permittivity(x, y)
            _eps_m = permittivity(x - dx, y + dy)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x - dx, y + dy)
            _eps_m = permittivity(x, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d[0] = (
            -a_tau_x * eps_p * n2_x * dx
            + b_x * n1_x * dx
            - a_x * eps_p * (3 - 2 * theta_l) / ((2 - theta_l) * (1 - theta_l))
        )
        d[1] = (
            a_tau_y * eps_p * n1_y * dy
            + b_y * n2_y * dy
            + a_y * eps_p * (3 - 2 * theta_t) / ((2 - theta_t) * (1 - theta_t))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M[0, 0] = (
            eps_p * (3 - 2 * theta_l) / ((1 - theta_l) * (2 - theta_l))
            + eps_m * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
            + eps_jump * n2_x**2 * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
        )
        M[0, 1] = eps_jump * n1_x * n2_x * dx / (dy * theta_t * (theta_t + 1))
        M[1, 0] = -eps_jump * n1_y * n2_y * dy / (dx * theta_l * (theta_l + 1))
        M[1, 1] = (
            -eps_p * (3 - 2 * theta_t) / ((1 - theta_t) * (2 - theta_t))
            - eps_m * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
            - eps_jump * n1_y**2 * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
        )

        # fmt: off
        N[0, offset(0, 0)] = (eps_m + eps_jump * n2_x**2) * (theta_l + 1) / theta_l \
            - (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_t * theta_l + theta_t - 1) / theta_t)
        N[0, offset(-1, 0)] = eps_p * (theta_l - 2) / (theta_l - 1)
        N[0, offset(-2, 0)] = -eps_p * (theta_l - 1) / (theta_l - 2)
        N[0, offset(1, 0)] = -(eps_m + eps_jump * n2_x**2) * theta_l / (theta_l + 1) \
            + eps_jump * n1_x * n2_x * theta_l * (dx / dy)
        N[0, offset(0, -1)] = eps_jump * n1_x * n2_x * (dx / dy) * (theta_t / (theta_t + 1) + theta_l) 
        N[0, offset(1, -1)] = -eps_jump * n1_x * n2_x * theta_l * (dx / dy)

        N[1, offset(0, 0)] = -(eps_m + eps_jump * n1_y**2) * (theta_t + 1) / theta_t \
            + (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_l * theta_t + theta_l - 1) / theta_l)
        N[1, offset(0, 1)] = -eps_p * (theta_t - 2) / (theta_t - 1)
        N[1, offset(0, 2)] = eps_p * (theta_t - 1) / (theta_t - 2)
        N[1, offset(0, -1)] = (eps_m + eps_jump * n1_y**2) * theta_t / (theta_t + 1) \
            - eps_jump * n1_y * n2_y * theta_t * (dy / dx)
        N[1, offset(1, 0)] = -eps_jump * n1_y * n2_y * (dy / dx) * (theta_l / (theta_l + 1) + theta_t)
        N[1, offset(1, -1)] = eps_jump * n1_y * n2_y * theta_t * (dy / dx)
        # fmt: on

        u_arr = np.array(
            [
                u[i + offset_x, j + offset_y]
                for offset_x in range(-2, 3)
                for offset_y in range(-2, 3)
            ]
        )
        u_I = np.linalg.solve(M, N @ u_arr + d)
        return (
            u_I[0],
            u[i + 1, j],
            u[i, j - 1],
            u_I[1],
            theta_l,
            theta_r,
            theta_b,
            theta_t,
        )
    elif direction == Direction.R | Direction.B:
        theta_r = compute_theta(Direction.R, i, j)
        theta_b = compute_theta(Direction.B, i, j)
        theta_l = 1.0
        theta_t = 1.0

        # normal evaluated at x_R and x_B
        n1_x = interp(Direction.R, theta_r, i, j, n1)
        n2_x = interp(Direction.R, theta_r, i, j, n2)
        n1_y = interp(Direction.B, theta_b, i, j, n1)
        n2_y = interp(Direction.B, theta_b, i, j, n2)

        # jump conditions at x_R and x_B
        a_x = interp(Direction.R, theta_r, i, j, a)
        a_y = interp(Direction.B, theta_b, i, j, a)
        b_x = interp(Direction.R, theta_r, i, j, b)
        b_y = interp(Direction.B, theta_b, i, j, b)

        # a_tau at x_R and x_B
        a_tau_x = interp(Direction.R, theta_r, i, j, a_tau)
        a_tau_y = interp(Direction.B, theta_b, i, j, a_tau)

        if eta > 0:
            _eps_p = permittivity(x, y)
            _eps_m = permittivity(x + dx, y - dy)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x + dx, y - dy)
            _eps_m = permittivity(x, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d[0] = (
            -a_tau_x * eps_p * n2_x * dx
            + b_x * n1_x * dx
            + a_x * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r))
        )
        d[1] = (
            a_tau_y * eps_p * n1_y * dy
            + b_y * n2_y * dy
            - a_y * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M[0, 0] = (
            -eps_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r))
            - eps_m * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
            - eps_jump * n2_x**2 * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
        )
        M[0, 1] = -eps_jump * n1_x * n2_x * dx / (dy * theta_b * (theta_b + 1))
        M[1, 0] = eps_jump * n1_y * n2_y * dy / (dx * theta_r * (theta_r + 1))
        M[1, 1] = (
            eps_p * (3 - 2 * theta_b) / ((1 - theta_b) * (2 - theta_b))
            + eps_m * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
            + eps_jump * n1_y**2 * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
        )

        # fmt: off
        N[0, offset(0, 0)] = -(eps_m + eps_jump * n2_x**2) * (theta_r + 1) / theta_r \
            + (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_b * theta_r + theta_b - 1) / theta_b)
        N[0, offset(1, 0)] = -eps_p * (theta_r - 2) / (theta_r - 1)
        N[0, offset(2, 0)] = eps_p * (theta_r - 1) / (theta_r - 2)
        N[0, offset(-1, 0)] = (eps_m + eps_jump * n2_x**2) * theta_r / (theta_r + 1) \
            - eps_jump * n1_x * n2_x * theta_r * (dx / dy)
        N[0, offset(0, 1)] = -eps_jump * n1_x * n2_x * (dx / dy) * (theta_b / (theta_b + 1) + theta_r) 
        N[0, offset(-1, 1)] = eps_jump * n1_x * n2_x * theta_r * (dx / dy)

        N[1, offset(0, 0)] = (eps_m + eps_jump * n1_y**2) * (theta_b + 1) / theta_b \
            - (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_r * theta_b + theta_r - 1) / theta_r)
        N[1, offset(0, -1)] = eps_p * (theta_b - 2) / (theta_b - 1)
        N[1, offset(0, -2)] = -eps_p * (theta_b - 1) / (theta_b - 2)
        N[1, offset(0, 1)] = -(eps_m + eps_jump * n1_y**2) * theta_b / (theta_b + 1) \
            + eps_jump * n1_y * n2_y * theta_b * (dy / dx)
        N[1, offset(-1, 0)] = eps_jump * n1_y * n2_y * (dy / dx) * (theta_r / (theta_r + 1) + theta_b)
        N[1, offset(-1, 1)] = -eps_jump * n1_y * n2_y * theta_b * (dy / dx)
        # fmt: on

        u_arr = np.array(
            [
                u[i + offset_x, j + offset_y]
                for offset_x in range(-2, 3)
                for offset_y in range(-2, 3)
            ]
        )
        u_I = np.linalg.solve(M, N @ u_arr + d)
        return (
            u[i - 1, j],
            u_I[0],
            u_I[1],
            u[i, j + 1],
            theta_l,
            theta_r,
            theta_b,
            theta_t,
        )
    elif direction == Direction.L | Direction.B:
        theta_l = compute_theta(Direction.L, i, j)
        theta_b = compute_theta(Direction.B, i, j)
        theta_r = 1.0
        theta_t = 1.0

        # normal evaluated at x_L and x_B
        n1_x = interp(Direction.L, theta_l, i, j, n1)
        n2_x = interp(Direction.L, theta_l, i, j, n2)
        n1_y = interp(Direction.B, theta_b, i, j, n1)
        n2_y = interp(Direction.B, theta_b, i, j, n2)

        # jump conditions at x_L and x_B
        a_x = interp(Direction.L, theta_l, i, j, a)
        a_y = interp(Direction.B, theta_b, i, j, a)
        b_x = interp(Direction.L, theta_l, i, j, b)
        b_y = interp(Direction.B, theta_b, i, j, b)

        # a_tau at x_L and x_B
        a_tau_x = interp(Direction.L, theta_l, i, j, a_tau)
        a_tau_y = interp(Direction.B, theta_b, i, j, a_tau)

        if eta > 0:
            _eps_p = permittivity(x, y)
            _eps_m = permittivity(x - dx, y - dy)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x - dx, y - dy)
            _eps_m = permittivity(x, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d[0] = (
            -a_tau_x * eps_p * n2_x * dx
            + b_x * n1_x * dx
            - a_x * eps_p * (3 - 2 * theta_l) / ((2 - theta_l) * (1 - theta_l))
        )
        d[1] = (
            a_tau_y * eps_p * n1_y * dy
            + b_y * n2_y * dy
            - a_y * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M[0, 0] = (
            eps_p * (3 - 2 * theta_l) / ((1 - theta_l) * (2 - theta_l))
            + eps_m * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
            + eps_jump * n2_x**2 * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
        )
        M[0, 1] = -eps_jump * n1_x * n2_x * dx / (dy * theta_b * (theta_b + 1))
        M[1, 0] = -eps_jump * n1_y * n2_y * dy / (dx * theta_l * (theta_l + 1))
        M[1, 1] = (
            eps_p * (3 - 2 * theta_b) / ((1 - theta_b) * (2 - theta_b))
            + eps_m * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
            + eps_jump * n1_y**2 * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
        )

        # fmt: off
        N[0, offset(0, 0)] = (eps_m + eps_jump * n2_x**2) * (theta_l + 1) / theta_l \
            + (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_b * theta_l + theta_b - 1) / theta_b)
        N[0, offset(-1, 0)] = eps_p * (theta_l - 2) / (theta_l - 1)
        N[0, offset(-2, 0)] = -eps_p * (theta_l - 1) / (theta_l - 2)
        N[0, offset(1, 0)] = -(eps_m + eps_jump * n2_x**2) * theta_l / (theta_l + 1) \
            - eps_jump * n1_x * n2_x * theta_l * (dx / dy)
        N[0, offset(0, 1)] = -eps_jump * n1_x * n2_x * (dx / dy) * (theta_b / (theta_b + 1) + theta_l) 
        N[0, offset(1, 1)] = eps_jump * n1_x * n2_x * theta_l * (dx / dy)

        N[1, offset(0, 0)] = (eps_m + eps_jump * n1_y**2) * (theta_b + 1) / theta_b \
            + (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_l * theta_b + theta_l - 1) / theta_l)
        N[1, offset(0, -1)] = eps_p * (theta_b - 2) / (theta_b - 1)
        N[1, offset(0, -2)] = -eps_p * (theta_b - 1) / (theta_b - 2)
        N[1, offset(0, 1)] = -(eps_m + eps_jump * n1_y**2) * theta_b / (theta_b + 1) \
            - eps_jump * n1_y * n2_y * theta_b * (dy / dx)
        N[1, offset(1, 0)] = -eps_jump * n1_y * n2_y * (dy / dx) * (theta_l / (theta_l + 1) + theta_b)
        N[1, offset(1, 1)] = eps_jump * n1_y * n2_y * theta_b * (dy / dx)
        # fmt: on

        u_arr = np.array(
            [
                u[i + offset_x, j + offset_y]
                for offset_x in range(-2, 3)
                for offset_y in range(-2, 3)
            ]
        )
        u_I = np.linalg.solve(M, N @ u_arr + d)
        return (
            u_I[0],
            u[i + 1, j],
            u_I[1],
            u[i, j + 1],
            theta_l,
            theta_r,
            theta_b,
            theta_t,
        )
    else:
        raise ValueError("Invalid direction for case 2", direction)


def convergence_test1():
    """
    [eps]=-1,
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

if __name__ == "__main__":
    # convergence_test1()
    convergence_test2()

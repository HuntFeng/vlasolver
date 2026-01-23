import enum

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

np.set_printoptions(legacy="1.25")  # no type info when printing


class Direction(enum.IntFlag):
    R = 1 << 0  # 0001
    T = 1 << 1  # 0010
    L = 1 << 2  # 0100
    B = 1 << 3  # 1000


def surface(x: float, y: float) -> float:
    pass


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


def index(i: int, j: int) -> int:
    """flatten index"""
    return i * ny + j


def center(i: int, j: int) -> tuple[float, float]:
    return x[i], y[j]


def compute_theta(direction: int, i: int, j: int) -> float:
    x, y = center(i, j)
    eta = surface(x, y)
    eta_r = surface(x + dx, y)
    eta_l = surface(x - dx, y)
    eta_t = surface(x, y + dy)

    eta_b = surface(x, y - dy)
    dx_eta = (eta_r - eta_l) / 2
    dy_eta = (eta_t - eta_b) / 2
    dxx_eta = (eta_r - 2 * eta + eta_l) / 2
    dyy_eta = (eta_t - 2 * eta + eta_b) / 2

    if direction == Direction.R:
        if np.isclose(dxx_eta, 0.0):
            theta = np.abs(eta / dx_eta)
        else:
            theta = (
                -dx_eta - np.sign(eta) * np.sqrt(dx_eta**2 - 4 * dxx_eta * eta)
            ) / (2 * dxx_eta)
    elif direction == Direction.T:
        if np.isclose(dyy_eta, 0.0):
            theta = np.abs(eta / dy_eta)
        else:
            theta = (
                -dy_eta - np.sign(eta) * np.sqrt(dy_eta**2 - 4 * dyy_eta * eta)
            ) / (2 * dyy_eta)
    elif direction == Direction.L:
        if np.isclose(dxx_eta, 0.0):
            theta = np.abs(eta / dx_eta)
        else:
            theta = (
                dx_eta - np.sign(eta) * np.sqrt(dx_eta**2 - 4 * dxx_eta * eta)
            ) / (2 * dxx_eta)
    elif direction == Direction.B:
        if np.isclose(dyy_eta, 0.0):
            theta = np.abs(eta / dy_eta)
        else:
            theta = (
                dy_eta - np.sign(eta) * np.sqrt(dy_eta**2 - 4 * dyy_eta * eta)
            ) / (2 * dyy_eta)
    else:
        theta = 1.0

    return theta


def interp(direction: int, theta: float, i: int, j: int, field: np.ndarray) -> float:
    """cubic interpolation"""
    t_matrix = np.array([1, theta, theta**2, theta**3])
    c_matrix = np.array([[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]])
    if direction == Direction.R:
        points = field[i - 1 : i + 3, j]
    elif direction == Direction.T:
        points = field[i, j - 1 : j + 3]
    elif direction == Direction.L:
        points = field[i - 2 : i + 2, j][::-1]
    elif direction == Direction.B:
        points = field[i, j - 2 : j + 2][::-1]
    else:
        raise ValueError("Invalid direction for interpolation", direction)
    val_I = 0.5 * t_matrix @ c_matrix @ points
    return val_I


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
    """coeff of u_ij and its neighbors for a case 1 cell"""
    x, y = center(i, j)
    row_idx = index(i, j)  # laplacian matrix row index
    eta = surface(x, y)  # assume this is negative for now
    theta = compute_theta(direction, i, j)
    a_tau_I = interp(direction, theta, i, j, a_tau)
    a_I = interp(direction, theta, i, j, a)
    b_I = interp(direction, theta, i, j, b)
    n1_I = interp(direction, theta, i, j, n1)
    n2_I = interp(direction, theta, i, j, n2)

    if direction == Direction.R:
        theta_l, theta_r, theta_t, theta_b = 1.0, theta, 1.0, 1.0
        # common denominator in discretization
        bot_x = (theta_r + theta_l) / 2 * dx**2
        bot_y = (theta_t + theta_b) / 2 * dy**2

        # permittivity
        eps_r = permittivity(x + theta_r * dx / 2, y)
        eps_l = permittivity(x - dx / 2, y)
        eps_t = permittivity(x, y + dy / 2)
        eps_b = permittivity(x, y - dy / 2)

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

        f[i, j] -= (d / M) * eps_r / theta_r / bot_x

        rows.extend([row_idx] * len(N))
        cols.extend(
            [
                index(i, j),
                index(i + 1, j),
                index(i + 2, j),
                index(i - 1, j),
                index(i, j - 1),
                index(i, j + 1),
                index(i - 1, j - 1),
            ]
        )
        vals.extend(
            [
                # u[i,j]
                (N[0] / M) * eps_r / theta_r / bot_x
                - (eps_r / theta_r + eps_l / theta_l) / bot_x
                - (eps_t / theta_t + eps_b / theta_b) / bot_y,
                # u[i+1,j]
                (N[1] / M) * eps_r / theta_r / bot_x,
                # u[i+2,j]
                (N[2] / M) * eps_r / theta_r / bot_x,
                # u[i-1,j]
                (N[3] / M) * eps_r / theta_r / bot_x + eps_l / theta_l / bot_x,
                # u[i,j-1]
                (N[4] / M) * eps_r / theta_r / bot_x + eps_b / theta_b / bot_y,
                # u[i,j+1]
                (N[5] / M) * eps_r / theta_r / bot_x + eps_t / theta_t / bot_y,
                # u_ext at [i-1,j-1]
                (N[6] / M) * eps_r / theta_r / bot_x,
            ]
        )
    elif direction == Direction.T:
        theta_l, theta_r, theta_t, theta_b = 1.0, 1.0, theta, 1.0

        bot_x = (theta_r + theta_l) / 2 * dx**2
        bot_y = (theta_t + theta_b) / 2 * dy**2

        eps_r = permittivity(x + dx / 2, y)
        eps_l = permittivity(x - dx / 2, y)
        eps_t = permittivity(x, y + theta_t * dy / 2)
        eps_b = permittivity(x, y - dy / 2)

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

        f[i, j] -= (d / M) * eps_t / theta_t / bot_y

        rows.extend([row_idx] * len(N))
        cols.extend(
            [
                index(i, j),
                index(i, j + 1),
                index(i, j + 2),
                index(i, j - 1),
                index(i - 1, j),
                index(i + 1, j),
                index(i - 1, j - 1),
            ]
        )
        vals.extend(
            [
                # u[i,j]
                (N[0] / M) * eps_t / theta_t / bot_y
                - (eps_r / theta_r + eps_l / theta_l) / bot_x
                - (eps_t / theta_t + eps_b / theta_b) / bot_y,
                # u[i,j+1]
                (N[1] / M) * eps_t / theta_t / bot_y,
                # u[i,j+2]
                (N[2] / M) * eps_t / theta_t / bot_y,
                # u[i,j-1]
                (N[3] / M) * eps_t / theta_t / bot_y + eps_b / theta_b / bot_y,
                # u[i-1,j]
                (N[4] / M) * eps_t / theta_t / bot_y + eps_l / theta_l / bot_x,
                # u[i+1,j]
                (N[5] / M) * eps_t / theta_t / bot_y + eps_r / theta_r / bot_x,
                # u_ext at [i-1,j-1]
                (N[6] / M) * eps_t / theta_t / bot_y,
            ]
        )
    elif direction == Direction.L:
        theta_l, theta_r, theta_t, theta_b = theta, 1.0, 1.0, 1.0

        bot_x = (theta_r + theta_l) / 2 * dx**2
        bot_y = (theta_t + theta_b) / 2 * dy**2

        eps_r = permittivity(x + dx / 2, y)
        eps_l = permittivity(x - theta_l * dx / 2, y)
        eps_t = permittivity(x, y + dy / 2)
        eps_b = permittivity(x, y - dy / 2)

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

        f[i, j] -= (d / M) * eps_l / theta_l / bot_x

        rows.extend([row_idx] * len(N))
        cols.extend(
            [
                index(i, j),
                index(i - 1, j),
                index(i - 2, j),
                index(i + 1, j),
                index(i, j - 1),
                index(i, j + 1),
                index(i + 1, j - 1),
            ]
        )
        vals.extend(
            [
                # u[i,j]
                (N[0] / M) * eps_l / theta_l / bot_x
                - (eps_r / theta_r + eps_l / theta_l) / bot_x
                - (eps_t / theta_t + eps_b / theta_b) / bot_y,
                # u[i-1,j]
                (N[1] / M) * eps_l / theta_l / bot_x,
                # u[i-2,j]
                (N[2] / M) * eps_l / theta_l / bot_x,
                # u[i+1,j]
                (N[3] / M) * eps_l / theta_l / bot_x + eps_r / theta_r / bot_x,
                # u[i,j+1]
                (N[4] / M) * eps_l / theta_l / bot_x + eps_t / theta_t / bot_y,
                # u[i,j-1]
                (N[5] / M) * eps_l / theta_l / bot_x + eps_b / theta_b / bot_y,
                # u_ext at [i+1,j-1]
                (N[6] / M) * eps_l / theta_l / bot_x,
            ]
        )
    elif direction == Direction.B:
        theta_l, theta_r, theta_t, theta_b = 1.0, 1.0, 1.0, theta

        bot_x = (theta_r + theta_l) / 2 * dx**2
        bot_y = (theta_t + theta_b) / 2 * dy**2

        eps_r = permittivity(x + dx / 2, y)
        eps_l = permittivity(x - dx / 2, y)
        eps_t = permittivity(x, y + dy / 2)
        eps_b = permittivity(x, y - theta_b * dy / 2)

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

        f[i, j] -= (d / M) * eps_b / theta_b / bot_y

        rows.extend([row_idx] * len(N))
        cols.extend(
            [
                index(i, j),
                index(i, j - 1),
                index(i, j - 2),
                index(i, j + 1),
                index(i - 1, j),
                index(i + 1, j),
                index(i - 1, j + 1),
            ]
        )
        vals.extend(
            [
                # u[i,j]
                (N[0] / M) * eps_b / theta_b / bot_y
                - (eps_r / theta_r + eps_l / theta_l) / bot_x
                - (eps_t / theta_t + eps_b / theta_b) / bot_y,
                # u[i,j-1]
                (N[1] / M) * eps_b / theta_b / bot_y,
                # u[i,j-2]
                (N[2] / M) * eps_b / theta_b / bot_y,
                # u[i,j+1]
                (N[3] / M) * eps_b / theta_b / bot_y + eps_t / theta_t / bot_y,
                # u[i-1,j]
                (N[4] / M) * eps_b / theta_b / bot_y + eps_l / theta_l / bot_x,
                # u[i+1,j]
                (N[5] / M) * eps_b / theta_b / bot_y + eps_r / theta_r / bot_x,
                # u_ext at [i-1,j+1]
                (N[6] / M) * eps_b / theta_b / bot_y,
            ]
        )
    else:
        raise ValueError("Invalid direction for case 1", direction)


def coeff_case2(direction: int, i: int, j: int):
    """coeff of u_ij and its neighbors for a case 2 cell"""
    x, y = center(i, j)
    eta = surface(x, y)
    row_idx = index(i, j)  # laplacian matrix row index

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

        bot_x = (theta_r + theta_l) / 2 * dx**2
        bot_y = (theta_t + theta_b) / 2 * dy**2

        eps_r = permittivity(x + theta_r * dx / 2, y)
        eps_l = permittivity(x - dx / 2, y)
        eps_t = permittivity(x, y + theta_t * dy / 2)
        eps_b = permittivity(x, y - dy / 2)

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

        M_inv = np.linalg.inv(M)
        M_inv_d = M_inv @ d
        M_inv_N = M_inv @ N

        f[i, j] -= (
            M_inv_d[0] * eps_r / theta_r / bot_x + M_inv_d[1] * eps_t / theta_t / bot_y
        )

        for offset_x in range(-2, 3):
            for offset_y in range(-2, 3):
                value = (
                    M_inv_N[0, offset(offset_x, offset_y)] * eps_r / theta_r / bot_x
                    + M_inv_N[1, offset(offset_x, offset_y)] * eps_t / theta_t / bot_y
                )
                if (offset_x, offset_y) == (0, 0):
                    value += (
                        -(eps_r / theta_r + eps_l / theta_l) / bot_x
                        - (eps_t / theta_t + eps_b / theta_b) / bot_y
                    )
                elif (offset_x, offset_y) == (-1, 0):
                    value += eps_l / theta_l / bot_x
                elif (offset_x, offset_y) == (0, -1):
                    value += eps_b / theta_b / bot_y
                rows.append(row_idx)
                cols.append(index(i + offset_x, j + offset_y))
                vals.append(value)

    elif direction == Direction.L | Direction.T:
        theta_l = compute_theta(Direction.L, i, j)
        theta_t = compute_theta(Direction.T, i, j)
        theta_r = 1.0
        theta_b = 1.0

        bot_x = (theta_r + theta_l) / 2 * dx**2
        bot_y = (theta_t + theta_b) / 2 * dy**2

        eps_r = permittivity(x + dx / 2, y)
        eps_l = permittivity(x - theta_l * dx / 2, y)
        eps_t = permittivity(x, y + theta_t * dy / 2)
        eps_b = permittivity(x, y - dy / 2)

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

        M_inv = np.linalg.inv(M)
        M_inv_d = M_inv @ d
        M_inv_N = M_inv @ N

        f[i, j] -= (
            M_inv_d[0] * eps_l / theta_l / bot_x + M_inv_d[1] * eps_t / theta_t / bot_y
        )

        for offset_x in range(-2, 3):
            for offset_y in range(-2, 3):
                value = (
                    M_inv_N[0, offset(offset_x, offset_y)] * eps_l / theta_l / bot_x
                    + M_inv_N[1, offset(offset_x, offset_y)] * eps_t / theta_t / bot_y
                )
                if (offset_x, offset_y) == (0, 0):
                    value += (
                        -(eps_r / theta_r + eps_l / theta_l) / bot_x
                        - (eps_t / theta_t + eps_b / theta_b) / bot_y
                    )
                elif (offset_x, offset_y) == (1, 0):
                    value += eps_r / theta_r / bot_x
                elif (offset_x, offset_y) == (0, -1):
                    value += eps_b / theta_b / bot_y

                rows.append(row_idx)
                cols.append(index(i + offset_x, j + offset_y))
                vals.append(value)

    elif direction == Direction.R | Direction.B:
        theta_r = compute_theta(Direction.R, i, j)
        theta_b = compute_theta(Direction.B, i, j)
        theta_l = 1.0
        theta_t = 1.0

        bot_x = (theta_r + theta_l) / 2 * dx**2
        bot_y = (theta_t + theta_b) / 2 * dy**2

        eps_r = permittivity(x + theta_r * dx / 2, y)
        eps_l = permittivity(x - dx / 2, y)
        eps_t = permittivity(x, y + dy / 2)
        eps_b = permittivity(x, y - theta_b * dy / 2)

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

        M_inv = np.linalg.inv(M)
        M_inv_d = M_inv @ d
        M_inv_N = M_inv @ N

        f[i, j] -= (
            M_inv_d[0] * eps_r / theta_r / bot_x + M_inv_d[1] * eps_b / theta_b / bot_y
        )

        for offset_x in range(-2, 3):
            for offset_y in range(-2, 3):
                value = (
                    M_inv_N[0, offset(offset_x, offset_y)] * eps_r / theta_r / bot_x
                    + M_inv_N[1, offset(offset_x, offset_y)] * eps_b / theta_b / bot_y
                )
                if (offset_x, offset_y) == (0, 0):
                    value += (
                        -(eps_r / theta_r + eps_l / theta_l) / bot_x
                        - (eps_t / theta_t + eps_b / theta_b) / bot_y
                    )
                elif (offset_x, offset_y) == (-1, 0):
                    value += eps_l / theta_l / bot_x
                elif (offset_x, offset_y) == (0, 1):
                    value += eps_t / theta_t / bot_y

                rows.append(row_idx)
                cols.append(index(i + offset_x, j + offset_y))
                vals.append(value)

    elif direction == Direction.L | Direction.B:
        theta_l = compute_theta(Direction.L, i, j)
        theta_b = compute_theta(Direction.B, i, j)
        theta_r = 1.0
        theta_t = 1.0

        bot_x = (theta_r + theta_l) / 2 * dx**2
        bot_y = (theta_t + theta_b) / 2 * dy**2

        eps_r = permittivity(x + dx / 2, y)
        eps_l = permittivity(x - theta_l * dx / 2, y)
        eps_t = permittivity(x, y + dy / 2)
        eps_b = permittivity(x, y - theta_b * dy / 2)

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

        M_inv = np.linalg.inv(M)
        M_inv_d = M_inv @ d
        M_inv_N = M_inv @ N

        f[i, j] -= (
            M_inv_d[0] * eps_l / theta_l / bot_x + M_inv_d[1] * eps_b / theta_b / bot_y
        )

        for offset_x in range(-2, 3):
            for offset_y in range(-2, 3):
                value = (
                    M_inv_N[0, offset(offset_x, offset_y)] * eps_l / theta_l / bot_x
                    + M_inv_N[1, offset(offset_x, offset_y)] * eps_b / theta_b / bot_y
                )
                if (offset_x, offset_y) == (0, 0):
                    value += (
                        -(eps_r / theta_r + eps_l / theta_l) / bot_x
                        - (eps_t / theta_t + eps_b / theta_b) / bot_y
                    )
                elif (offset_x, offset_y) == (1, 0):
                    value += eps_r / theta_r / bot_x
                elif (offset_x, offset_y) == (0, 1):
                    value += eps_t / theta_t / bot_y

                rows.append(row_idx)
                cols.append(index(i + offset_x, j + offset_y))
                vals.append(value)

    else:
        raise ValueError("Invalid direction for case 2", direction)


def construct_matrix():
    for i in range(nx):
        for j in range(ny):
            # if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            if i < 2 or i >= nx - 2 or j < 2 or j >= ny - 2:
                rows.append(index(i, j))
                cols.append(index(i, j))
                vals.append(1.0)
                # print(u_exact[i, j])
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
                    coeff_case2(direction, i, j)
                case _:
                    raise NotImplementedError("More than 2 cuts not implemented yet.")

    A = coo_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    return A


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
        M_inv = np.linalg.inv(M)
        u_I = M_inv @ (N @ u_arr + d)
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
        M_inv = np.linalg.inv(M)
        u_I = M_inv @ (N @ u_arr + d)
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
        M_inv = np.linalg.inv(M)
        u_I = M_inv @ (N @ u_arr + d)
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
        M_inv = np.linalg.inv(M)
        u_I = M_inv @ (N @ u_arr + d)
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
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                        interface_value_case2(direction, i, j, u)
                    )
                case _:
                    raise NotImplementedError("More than 2 cuts not implemented yet.")

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

    n_range = 2 ** np.arange(4, 8, dtype=int)
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

    n_range = 2 ** np.arange(3, 9, dtype=int)
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
        # n1, n2 = X / np.sqrt(X**2 + Y**2), Y / np.sqrt(X**2 + Y**2)
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

    # convergence table
    print("Convergence Table:")
    print(f" N    Err_u({norm})  Order Err_du({norm}) Order")
    print("-------------------------")
    for i, n in enumerate(n_range):
        if i == 0:
            order_u = np.nan
            order_du = np.nan
        else:
            order_u = np.log(errors_u[i - 1] / errors_u[i]) / np.log(2)
            order_du = np.log(errors_du[i - 1] / errors_du[i]) / np.log(2)
        print(f"{n} {errors_u[i]:.2e} {order_u:.2f} {errors_du[i]:.2e} {order_du:.2f}")
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
    plt.loglog(1 / n_range, 1 / n_range, "--", label="$O(h)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $u$")
    plt.subplot(122)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.loglog(1 / n_range, 1 / n_range, "--", label="$O(h)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $\\nabla u$")

    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(X, Y, u_exact, shading="auto")
    plt.colorbar()
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(122)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar()
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")

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

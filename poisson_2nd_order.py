import enum

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

np.set_printoptions(legacy="1.25")  # no type info when printing

M_EPS = 1e-6  # a small number


class Direction(enum.IntFlag):
    R = 1 << 0  # 0001
    T = 1 << 1  # 0010
    L = 1 << 2  # 0100
    B = 1 << 3  # 1000


def surface(x: float, y: float) -> float:
    pass
    # xI = 0.52
    # # return x - 2.0
    # return x - xI
    # return -(x - xI)
    # return x**2 - xI**2
    # return -(x**2 - xI**2)
    # return y - xI
    # return -(y - xI)
    # return y**2 - xI**2
    # return -(y**2 - xI**2)


def normal(x: float, y: float) -> tuple[float, float]:
    # default should be 4th order centrl difference

    # dx_eta = 2 * (x - 0.5)
    # dy_eta = 2 * (y - 0.5)
    # norm = np.sqrt(dx_eta**2 + dy_eta**2)
    # return dx_eta / norm, dy_eta / norm
    # return 1.0, 0.0
    # return 0.0, 1.0
    pass


def permittivity(x: float, y: float) -> float:
    # return 1000.0 if surface(x, y) < 0 else 1.0
    # return 1.0
    pass


def index(i: int, j: int) -> int:
    """flatten index"""
    # if i * ny + j > nx * ny:
    #     breakpoint()
    return i * ny + j


def center(i: int, j: int) -> tuple[float, float]:
    x = i * dx + dx / 2
    y = j * dy + dy / 2
    return x, y


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

    if direction == Direction.R or direction == Direction.L:
        if np.isclose(dxx_eta, 0.0):
            theta = np.abs(eta / dx_eta)
        else:
            theta = (
                -dx_eta - np.sign(eta) * np.sqrt(dx_eta**2 - 4 * dxx_eta * eta)
            ) / (2 * dxx_eta)
    elif direction == Direction.T or direction == Direction.B:
        if np.isclose(dyy_eta, 0.0):
            theta = np.abs(eta / dy_eta)
        else:
            theta = (
                -dy_eta - np.sign(eta) * np.sqrt(dy_eta**2 - 4 * dyy_eta * eta)
            ) / (2 * dyy_eta)
    # elif direction == Direction.L:
    #     if np.isclose(dxx_eta, 0.0):
    #         theta = np.abs(eta / dx_eta)
    #     else:
    #         theta = (
    #             -dx_eta
    #             + np.sign(eta) * np.sqrt(dx_eta**2 - 4 * dxx_eta * eta)
    #         ) / (2 * dxx_eta)
    # elif direction == Direction.B:
    #     if np.isclose(dyy_eta, 0.0):
    #         theta = np.abs(eta / dy_eta)
    #     else:
    #         theta = (
    #             -dy_eta
    #             + np.sign(eta) * np.sqrt(dy_eta**2 - 4 * dyy_eta * eta)
    #         ) / (2 * dyy_eta)
    else:
        theta = 1.0

    return theta


def compute_a_tau(i: int, j: int) -> float:
    """Compute tangential derivative of jump condition a at (i, j)"""
    # TODO: if i+2 or j+2 exceeds
    if i + 2 > nx - 1 or i - 2 < 0 or j + 2 > ny - 1 or j - 2 < 0:
        dx_a = (a[i + 1, j] - a[i - 1, j]) / (2 * dx)
        dy_a = (a[i, j + 1] - a[i, j - 1]) / (2 * dy)
    else:
        dx_a = (-a[i + 2, j] + 8 * a[i + 1, j] - 8 * a[i - 1, j] + a[i - 2, j]) / (
            12 * dx
        )
        dy_a = (-a[i, j + 2] + 8 * a[i, j + 1] - 8 * a[i, j - 1] + a[i, j - 2]) / (
            12 * dy
        )
    n1, n2 = normal(*center(i, j))
    a_tau = -dx_a * n2 + dy_a * n1
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
    n1, n2 = normal(x, y)
    a_tau = compute_a_tau(i, j)
    theta = compute_theta(direction, i, j)

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
            _eps_p = permittivity(x + theta_r * dx - M_EPS, y)
            _eps_m = permittivity(x + theta_r * dx + M_EPS, y)
            eps_jump = _eps_p - _eps_m
            # swap these two variable in the d expression
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x + theta_r * dx + M_EPS, y)
            _eps_m = permittivity(x + theta_r * dx - M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            -a_tau * eps_p * n2 * dx
            + b[i, j] * n1 * dx
            + a[i, j] * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r))
        )

        if eta > 0:
            # in the following formulas, permittivity signs are also swapped
            # the permittivity jump stays the same
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            -eps_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r))
            - eps_m * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
            - eps_jump * n2**2 * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
        )

        N = [
            # u[i,j]
            -eps_jump * n1 * n2 * theta_r * dx / dy
            - (eps_jump * n2**2 + eps_m) * (1 + theta_r) / theta_r,
            # u[i+1,j]
            -eps_p * (theta_r - 2) / (theta_r - 1),
            # u[i+2,j]
            eps_p * (theta_r - 1) / (theta_r - 2),
            # u[i-1,j]
            eps_jump * n1 * n2 * theta_r * dx / dy
            + (eps_jump * n2**2 + eps_m) * theta_r / (1 + theta_r),
            # u[i,j-1]
            eps_jump * n1 * n2 * (2 * theta_r + 1) * dx / (2 * dy),
            # u[i,j+1]
            -eps_jump * n1 * n2 * dx / (2 * dy),
            # u[i-1,j-1]
            -eps_jump * n1 * n2 * theta_r * dx / dy,
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
            _eps_p = permittivity(x, y + theta_t * dy - M_EPS)
            _eps_m = permittivity(x, y + theta_t * dy + M_EPS)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x, y + theta_t * dy + M_EPS)
            _eps_m = permittivity(x, y + theta_t * dy - M_EPS)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            a_tau * eps_p * n1 * dy
            + b[i, j] * n2 * dy
            + a[i, j] * eps_p * (3 - 2 * theta_t) / ((2 - theta_t) * (1 - theta_t))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            -eps_p * (3 - 2 * theta_t) / ((1 - theta_t) * (2 - theta_t))
            - eps_m * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
            - eps_jump * n1**2 * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
        )
        N = [
            # u[i,j]
            -eps_jump * n1 * n2 * theta_t * dy / dx
            - (eps_jump * n1**2 + eps_m) * (1 + theta_t) / theta_t,
            # u[i,j+1]
            -eps_p * (theta_t - 2) / (theta_t - 1),
            # u[i,j+2]
            eps_p * (theta_t - 1) / (theta_t - 2),
            # u[i,j-1]
            eps_jump * n1 * n2 * theta_t * dy / dx
            + (eps_jump * n1**2 + eps_m) * theta_t / (1 + theta_t),
            # u[i-1,j]
            eps_jump * n1 * n2 * (2 * theta_t + 1) * dy / (2 * dx),
            # u[i+1,j]
            -eps_jump * n1 * n2 * dy / (2 * dx),
            # u[i-1,j-1]
            -eps_jump * n1 * n2 * theta_t * dy / dx,
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
            _eps_p = permittivity(x - theta_l * dx + M_EPS, y)
            _eps_m = permittivity(x - theta_l * dx - M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x - theta_l * dx - M_EPS, y)
            _eps_m = permittivity(x - theta_l * dx + M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            -a_tau * eps_p * n2 * dx
            + b[i, j] * n1 * dx
            - a[i, j] * eps_p * (3 - 2 * theta_l) / ((2 - theta_l) * (1 - theta_l))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            eps_p * (3 - 2 * theta_l) / ((1 - theta_l) * (2 - theta_l))
            + eps_m * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
            + eps_jump * n2**2 * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
        )

        N = [
            # u[i,j]
            -eps_jump * n1 * n2 * theta_l * dx / dy
            + (eps_jump * n2**2 + eps_m) * (1 + theta_l) / theta_l,
            # u[i-1,j]
            eps_p * (theta_l - 2) / (theta_l - 1),
            # u[i-2,j]
            -eps_p * (theta_l - 1) / (theta_l - 2),
            # u[i+1,j]
            eps_jump * n1 * n2 * theta_l * dx / dy
            - (eps_jump * n2**2 + eps_m) * theta_l / (1 + theta_l),
            # u[i,j-1]
            eps_jump * n1 * n2 * (2 * theta_l + 1) * dx / (2 * dy),
            # u[i,j+1]
            -eps_jump * n1 * n2 * dx / (2 * dy),
            # u[i+1,j-1]
            -eps_jump * n1 * n2 * theta_l * dx / dy,
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
            _eps_p = permittivity(x, y - theta_b * dy + M_EPS)
            _eps_m = permittivity(x, y - theta_b * dy - M_EPS)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x, y - theta_b * dy - M_EPS)
            _eps_m = permittivity(x, y - theta_b * dy + M_EPS)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            a_tau * eps_p * n1 * dy
            + b[i, j] * n2 * dy
            - a[i, j] * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            eps_p * (3 - 2 * theta_b) / ((1 - theta_b) * (2 - theta_b))
            + eps_m * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
            + eps_jump * n1**2 * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
        )
        # N = [
        #     # u[i,j]
        #     eps_jump * n1 * n2 * theta_b * dy / dx
        #     + (eps_jump * n1**2 + eps_m) * (1 + theta_b) / theta_b,
        #     # u[i,j-1]
        #     eps_p * (theta_b - 2) / (theta_b - 1),
        #     # u[i,j-2]
        #     -eps_p * (theta_b - 1) / (theta_b - 2),
        #     # u[i,j+1]
        #     -eps_jump * n1 * n2 * theta_b * dy / dx
        #     - (eps_jump * n1**2 + eps_m) * theta_b / (1 + theta_b),
        #     # u[i-1,j]
        #     eps_jump * n1 * n2 * dy / (2 * dx),
        #     # u[i+1,j]
        #     -eps_jump * n1 * n2 * (2 * theta_b + 1) * dy / (2 * dx),
        #     # u[i+1,j+1]
        #     eps_jump * n1 * n2 * theta_b * dy / dx,
        # ]
        N = [
            # u[i,j]
            -eps_jump * n1 * n2 * theta_b * dy / dx
            + (eps_jump * n1**2 + eps_m) * (1 + theta_b) / theta_b,
            # u[i,j-1]
            eps_p * (theta_b - 2) / (theta_b - 1),
            # u[i,j-2]
            -eps_p * (theta_b - 1) / (theta_b - 2),
            # u[i,j+1]
            eps_jump * n1 * n2 * theta_b * dy / dx
            - (eps_jump * n1**2 + eps_m) * theta_b / (1 + theta_b),
            # u[i-1,j]
            eps_jump * n1 * n2 * (2 * theta_b + 1) * dy / (2 * dx),
            # u[i+1,j]
            -eps_jump * n1 * n2 * dy / (2 * dx),
            # u[i-1,j+1]
            -eps_jump * n1 * n2 * theta_b * dy / dx,
        ]

        f[i, j] -= (d / M) * eps_b / theta_b / bot_y

        rows.extend([row_idx] * len(N))
        # cols.extend(
        #     [
        #         index(i, j),
        #         index(i, j - 1),
        #         index(i, j - 2),
        #         index(i, j + 1),
        #         index(i - 1, j),
        #         index(i + 1, j),
        #         index(i + 1, j + 1),
        #     ]
        # )
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
    # calculate M = [[xx, xx], [xx, xx]] # 2x2 matrix
    # calculate N = [[xx, xx, ...], [xx, xx, ...]] # 2xn_neighbor matrix
    # calculate d = [[xx], [xx]] # 2x1 matrix
    # return M_inv@N, M_inv@d

    x, y = center(i, j)
    eta = surface(x, y)
    a_tau = compute_a_tau(i, j)
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
        n1_x, n2_x = normal(x + theta_r * dx - M_EPS, y)
        n1_y, n2_y = normal(x, y + theta_t * dy - M_EPS)

        if eta > 0:
            _eps_p = permittivity(x + theta_r * dx - M_EPS, y)
            _eps_m = permittivity(x + theta_r * dx + M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x + theta_r * dx + M_EPS, y)
            _eps_m = permittivity(x + theta_r * dx - M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d[0] = (
            -a_tau * eps_p * n2_x * dx
            + b[i, j] * n1_x * dx
            + a[i, j] * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r))
        )
        d[1] = (
            a_tau * eps_p * n1_y * dy
            + b[i, j] * n2_y * dx
            + a[i, j] * eps_p * (3 - 2 * theta_t) / ((2 - theta_t) * (1 - theta_t))
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
            - eps_jump * n2_y**2 * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
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
        N[0, offset(-1, 0)] = (eps_m + eps_jump * n2_x**2) / (theta_r + 1) \
            + eps_jump * n1_x * n2_x * dx / dy
        # u[i,j-1]
        N[0, offset(0, -1)] = eps_jump * n1_x * n2_x * (dx / dy) * (theta_t / (theta_t + 1) + theta_r) 
        # u[i-1,j-1]
        N[0, offset(-1, -1)] = -eps_jump * n1_x * n2_x * theta_r * (dx / dy)

        # u[i,j]
        N[1, offset(0,0)] = -(eps_m + eps_jump * n1_y**2) * (theta_t + 1) / theta_t \
            - (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_r * theta_t + theta_r - 1) / theta_r)
        # u[i,j+1]
        N[1, offset(0,1)] = -eps_p * (theta_t - 2) / (theta_t - 1)
        # u[i,j+2]
        N[1, offset(0,2)] = eps_p * (theta_t - 1) / (theta_t - 2)
        # u[i,j-1]
        N[1, offset(0,-1)] = (eps_m + eps_jump * n1_y**2) / (theta_t + 1) \
            + eps_jump * n1_y * n2_y * dy / dx
        # u[i-1,j]
        N[1, offset(-1,0)] = eps_jump * n1_y * n2_y * (dy / dx) * (theta_r / (theta_r + 1) + theta_t)
        # u[i-1,j-1]
        N[1, offset(-1,-1)] = -eps_jump * n1_y * n2_y * theta_t * (dy / dx)
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
        n1_x, n2_x = normal(x - theta_l * dx + M_EPS, y)
        n1_y, n2_y = normal(x, y + theta_t * dy - M_EPS)

        if eta > 0:
            _eps_p = permittivity(x - theta_l * dx + M_EPS, y)
            _eps_m = permittivity(x - theta_l * dx - M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x - theta_l * dx - M_EPS, y)
            _eps_m = permittivity(x - theta_l * dx + M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d[0] = (
            -a_tau * eps_p * n2_x * dx
            + b[i, j] * n1_x * dx
            - a[i, j] * eps_p * (3 - 2 * theta_l) / ((2 - theta_l) * (1 - theta_l))
        )
        d[1] = (
            a_tau * eps_p * n1_y * dy
            + b[i, j] * n2_y * dx
            + a[i, j] * eps_p * (3 - 2 * theta_t) / ((2 - theta_t) * (1 - theta_t))
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
            - eps_jump * n2_y**2 * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
        )

        # fmt: off
        N[0, offset(0, 0)] = (eps_m + eps_jump * n2_x**2) * (theta_l + 1) / theta_l \
            - (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_t * theta_l + theta_t - 1) / theta_t)
        N[0, offset(-1, 0)] = eps_p * (theta_l - 2) / (theta_l - 1)
        N[0, offset(-2, 0)] = -eps_p * (theta_l - 1) / (theta_l - 2)
        N[0, offset(1, 0)] = -(eps_m + eps_jump * n2_x**2) / (theta_l + 1) \
            + eps_jump * n1_x * n2_x * dx / dy
        N[0, offset(0, -1)] = eps_jump * n1_x * n2_x * (dx / dy) * (theta_t / (theta_t + 1) + theta_l) 
        N[0, offset(1, -1)] = -eps_jump * n1_x * n2_x * theta_l * (dx / dy)

        N[1, offset(0,0)] = -(eps_m + eps_jump * n1_y**2) * (theta_t + 1) / theta_t \
            + (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_l * theta_t + theta_l - 1) / theta_l)
        N[1, offset(0,1)] = -eps_p * (theta_t - 2) / (theta_t - 1)
        N[1, offset(0,2)] = eps_p * (theta_t - 1) / (theta_t - 2)
        N[1, offset(0,-1)] = (eps_m + eps_jump * n1_y**2) / (theta_t + 1) \
            - eps_jump * n1_y * n2_y * dy / dx
        N[1, offset(1,0)] = -eps_jump * n1_y * n2_y * (dy / dx) * (theta_l / (theta_l + 1) + theta_t)
        N[1, offset(1,-1)] = eps_jump * n1_y * n2_y * theta_t * (dy / dx)
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
        n1_x, n2_x = normal(x + theta_r * dx - M_EPS, y)
        n1_y, n2_y = normal(x, y - theta_b * dy + M_EPS)

        if eta > 0:
            _eps_p = permittivity(x + theta_r * dx - M_EPS, y)
            _eps_m = permittivity(x + theta_r * dx + M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x + theta_r * dx + M_EPS, y)
            _eps_m = permittivity(x + theta_r * dx - M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d[0] = (
            -a_tau * eps_p * n2_x * dx
            + b[i, j] * n1_x * dx
            + a[i, j] * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r))
        )
        d[1] = (
            a_tau * eps_p * n1_y * dy
            + b[i, j] * n2_y * dx
            - a[i, j] * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b))
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
            + eps_jump * n2_y**2 * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
        )

        # fmt: off
        N[0, offset(0, 0)] = -(eps_m + eps_jump * n2_x**2) * (theta_r + 1) / theta_r \
            + (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_b * theta_r + theta_b - 1) / theta_b)
        N[0, offset(1, 0)] = -eps_p * (theta_r - 2) / (theta_r - 1)
        N[0, offset(2, 0)] = eps_p * (theta_r - 1) / (theta_r - 2)
        N[0, offset(-1, 0)] = (eps_m + eps_jump * n2_x**2) / (theta_r + 1) \
            - eps_jump * n1_x * n2_x * dx / dy
        N[0, offset(0, 1)] = -eps_jump * n1_x * n2_x * (dx / dy) * (theta_b / (theta_b + 1) + theta_r) 
        N[0, offset(-1, 1)] = eps_jump * n1_x * n2_x * theta_r * (dx / dy)

        N[1, offset(0,0)] = (eps_m + eps_jump * n1_y**2) * (theta_b + 1) / theta_b \
            - (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_r * theta_b + theta_r - 1) / theta_r)
        N[1, offset(0,-1)] = eps_p * (theta_b - 2) / (theta_b - 1)
        N[1, offset(0,2)] = -eps_p * (theta_b - 1) / (theta_b - 2)
        N[1, offset(0,1)] = -(eps_m + eps_jump * n1_y**2) / (theta_b + 1) \
            + eps_jump * n1_y * n2_y * dy / dx
        N[1, offset(-1,0)] = eps_jump * n1_y * n2_y * (dy / dx) * (theta_r / (theta_r + 1) + theta_b)
        N[1, offset(-1,1)] = -eps_jump * n1_y * n2_y * theta_b * (dy / dx)
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
        n1_x, n2_x = normal(x - theta_l * dx + M_EPS, y)
        n1_y, n2_y = normal(x, y - theta_b * dy + M_EPS)

        if eta > 0:
            _eps_p = permittivity(x - theta_l * dx + M_EPS, y)
            _eps_m = permittivity(x - theta_l * dx - M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x - theta_l * dx - M_EPS, y)
            _eps_m = permittivity(x - theta_l * dx + M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d[0] = (
            -a_tau * eps_p * n2_x * dx
            + b[i, j] * n1_x * dx
            - a[i, j] * eps_p * (3 - 2 * theta_l) / ((2 - theta_l) * (1 - theta_l))
        )
        d[1] = (
            a_tau * eps_p * n1_y * dy
            + b[i, j] * n2_y * dx
            - a[i, j] * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b))
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
            + eps_jump * n2_y**2 * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
        )

        # fmt: off
        N[0, offset(0, 0)] = (eps_m + eps_jump * n2_x**2) * (theta_l + 1) / theta_l \
            + (eps_jump * n1_x * n2_x) * (dx / dy) * ((theta_b * theta_l + theta_b - 1) / theta_b)
        N[0, offset(-1, 0)] = eps_p * (theta_l - 2) / (theta_l - 1)
        N[0, offset(-2, 0)] = -eps_p * (theta_l - 1) / (theta_l - 2)
        N[0, offset(1, 0)] = -(eps_m + eps_jump * n2_x**2) / (theta_l + 1) \
            - eps_jump * n1_x * n2_x * dx / dy
        N[0, offset(0, 1)] = -eps_jump * n1_x * n2_x * (dx / dy) * (theta_b / (theta_b + 1) + theta_l) 
        N[0, offset(1, 1)] = eps_jump * n1_x * n2_x * theta_l * (dx / dy)

        N[1, offset(0,0)] = (eps_m + eps_jump * n1_y**2) * (theta_b + 1) / theta_b \
            + (eps_jump * n1_y * n2_y) * (dy / dx) * ((theta_l * theta_b + theta_l - 1) / theta_l)
        N[1, offset(0,-1)] = eps_p * (theta_b - 2) / (theta_b - 1)
        N[1, offset(0,-2)] = -eps_p * (theta_b - 1) / (theta_b - 2)
        N[1, offset(0,1)] = -(eps_m + eps_jump * n1_y**2) / (theta_b + 1) \
            - eps_jump * n1_y * n2_y * dy / dx
        N[1, offset(1,0)] = -eps_jump * n1_y * n2_y * (dy / dx) * (theta_l / (theta_l + 1) + theta_b)
        N[1, offset(1,1)] = eps_jump * n1_y * n2_y * theta_b * (dy / dx)
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
        raise ValueError("Invalid direction for case 1", direction)


def construct_matrix():
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
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
                    coeff_case2(direction, i, j)
                case _:
                    raise NotImplementedError("More than 2 cuts not implemented yet.")

    A = coo_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    return A

def interface_value_case0(i: int, j: int, u: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    return u[i-1,j], u[i+1,j], u[i,j-1], u[i, j+1], 1.0, 1.0, 1.0, 1.0

def interface_value_case1(direction: int, i: int, j: int, u: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    x, y = center(i, j)
    eta = surface(x, y)  # assume this is negative for now
    n1, n2 = normal(x, y)
    a_tau = compute_a_tau(i, j)
    theta = compute_theta(direction, i, j)

    if direction == Direction.R:
        theta_l, theta_r, theta_b, theta_t = 1.0, theta, 1.0, 1.0

        if eta > 0:
            _eps_p = permittivity(x + theta_r * dx - M_EPS, y)
            _eps_m = permittivity(x + theta_r * dx + M_EPS, y)
            eps_jump = _eps_p - _eps_m
            # swap these two variable in the d expression
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x + theta_r * dx + M_EPS, y)
            _eps_m = permittivity(x + theta_r * dx - M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            -a_tau * eps_p * n2 * dx
            + b[i, j] * n1 * dx
            + a[i, j] * eps_p * (3 - 2 * theta_r) / ((2 - theta_r) * (1 - theta_r))
        )

        if eta > 0:
            # in the following formulas, permittivity signs are also swapped
            # the permittivity jump stays the same
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            -eps_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r))
            - eps_m * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
            - eps_jump * n2**2 * (2 * theta_r + 1) / (theta_r * (theta_r + 1))
        )

        N = [
            # u[i,j]
            -eps_jump * n1 * n2 * theta_r * dx / dy
            - (eps_jump * n2**2 + eps_m) * (1 + theta_r) / theta_r,
            # u[i+1,j]
            -eps_p * (theta_r - 2) / (theta_r - 1),
            # u[i+2,j]
            eps_p * (theta_r - 1) / (theta_r - 2),
            # u[i-1,j]
            eps_jump * n1 * n2 * theta_r * dx / dy
            + (eps_jump * n2**2 + eps_m) * theta_r / (1 + theta_r),
            # u[i,j-1]
            eps_jump * n1 * n2 * (2 * theta_r + 1) * dx / (2 * dy),
            # u[i,j+1]
            -eps_jump * n1 * n2 * dx / (2 * dy),
            # u[i-1,j-1]
            -eps_jump * n1 * n2 * theta_r * dx / dy,
        ]
        u_arr = [u[i,j], u[i+1,j], u[i+2,j], u[i-1,j], u[i,j-1], u[i,j+1], u[i-1,j-1],]
        u_I = (np.dot(N, u_arr) + d)/M
        return u[i-1,j], u_I, u[i,j-1], u[i, j+1], theta_l, theta_r, theta_b, theta_t
    elif direction == Direction.T:
        theta_l, theta_r, theta_b, theta_t = 1.0, 1.0, 1.0, theta

        if eta > 0:
            _eps_p = permittivity(x, y + theta_t * dy - M_EPS)
            _eps_m = permittivity(x, y + theta_t * dy + M_EPS)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x, y + theta_t * dy + M_EPS)
            _eps_m = permittivity(x, y + theta_t * dy - M_EPS)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            a_tau * eps_p * n1 * dy
            + b[i, j] * n2 * dy
            + a[i, j] * eps_p * (3 - 2 * theta_t) / ((2 - theta_t) * (1 - theta_t))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            -eps_p * (3 - 2 * theta_t) / ((1 - theta_t) * (2 - theta_t))
            - eps_m * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
            - eps_jump * n1**2 * (2 * theta_t + 1) / (theta_t * (theta_t + 1))
        )
        N = [
            # u[i,j]
            -eps_jump * n1 * n2 * theta_t * dy / dx
            - (eps_jump * n1**2 + eps_m) * (1 + theta_t) / theta_t,
            # u[i,j+1]
            -eps_p * (theta_t - 2) / (theta_t - 1),
            # u[i,j+2]
            eps_p * (theta_t - 1) / (theta_t - 2),
            # u[i,j-1]
            eps_jump * n1 * n2 * theta_t * dy / dx
            + (eps_jump * n1**2 + eps_m) * theta_t / (1 + theta_t),
            # u[i-1,j]
            eps_jump * n1 * n2 * (2 * theta_t + 1) * dy / (2 * dx),
            # u[i+1,j]
            -eps_jump * n1 * n2 * dy / (2 * dx),
            # u[i-1,j-1]
            -eps_jump * n1 * n2 * theta_t * dy / dx,
        ]
        u_arr = [u[i,j], u[i,j+1], u[i,j+2], u[i,j-1], u[i-1,j], u[i+1,j], u[i-1,j-1],]
        u_I = (np.dot(N, u_arr) + d)/M
        return u[i-1,j], u[i+1,j], u[i,j-1], u_I, theta_l, theta_r, theta_b, theta_t
    elif direction == Direction.L:
        theta_l, theta_r, theta_b, theta_t = theta, 1.0, 1.0, 1.0

        if eta > 0:
            _eps_p = permittivity(x - theta_l * dx + M_EPS, y)
            _eps_m = permittivity(x - theta_l * dx - M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x - theta_l * dx - M_EPS, y)
            _eps_m = permittivity(x - theta_l * dx + M_EPS, y)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            -a_tau * eps_p * n2 * dx
            + b[i, j] * n1 * dx
            - a[i, j] * eps_p * (3 - 2 * theta_l) / ((2 - theta_l) * (1 - theta_l))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            eps_p * (3 - 2 * theta_l) / ((1 - theta_l) * (2 - theta_l))
            + eps_m * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
            + eps_jump * n2**2 * (2 * theta_l + 1) / (theta_l * (theta_l + 1))
        )

        N = [
            # u[i,j]
            -eps_jump * n1 * n2 * theta_l * dx / dy
            + (eps_jump * n2**2 + eps_m) * (1 + theta_l) / theta_l,
            # u[i-1,j]
            eps_p * (theta_l - 2) / (theta_l - 1),
            # u[i-2,j]
            -eps_p * (theta_l - 1) / (theta_l - 2),
            # u[i+1,j]
            eps_jump * n1 * n2 * theta_l * dx / dy
            - (eps_jump * n2**2 + eps_m) * theta_l / (1 + theta_l),
            # u[i,j-1]
            eps_jump * n1 * n2 * (2 * theta_l + 1) * dx / (2 * dy),
            # u[i,j+1]
            -eps_jump * n1 * n2 * dx / (2 * dy),
            # u[i+1,j-1]
            -eps_jump * n1 * n2 * theta_l * dx / dy,
        ]
        u_arr = [u[i,j], u[i-1,j], u[i-2,j], u[i+1,j], u[i,j-1], u[i,j+1], u[i+1,j-1],]
        u_I = (np.dot(N, u_arr) + d)/M
        return u_I, u[i+1,j], u[i, j-1], u[i, j+1], theta_l, theta_r, theta_b, theta_t
    elif direction == Direction.B:
        theta_l, theta_r, theta_b, theta_t = 1.0, 1.0, theta, 1.0

        if eta > 0:
            _eps_p = permittivity(x, y - theta_b * dy + M_EPS)
            _eps_m = permittivity(x, y - theta_b * dy - M_EPS)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_m, _eps_p
        else:
            _eps_p = permittivity(x, y - theta_b * dy - M_EPS)
            _eps_m = permittivity(x, y - theta_b * dy + M_EPS)
            eps_jump = _eps_p - _eps_m
            eps_p, eps_m = _eps_p, _eps_m

        d = (
            a_tau * eps_p * n1 * dy
            + b[i, j] * n2 * dy
            - a[i, j] * eps_p * (3 - 2 * theta_b) / ((2 - theta_b) * (1 - theta_b))
        )

        if eta > 0:
            eps_p, eps_m = -_eps_m, -_eps_p

        M = (
            eps_p * (3 - 2 * theta_b) / ((1 - theta_b) * (2 - theta_b))
            + eps_m * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
            + eps_jump * n1**2 * (2 * theta_b + 1) / (theta_b * (theta_b + 1))
        )
        N = [
            # u[i,j]
            -eps_jump * n1 * n2 * theta_b * dy / dx
            + (eps_jump * n1**2 + eps_m) * (1 + theta_b) / theta_b,
            # u[i,j-1]
            eps_p * (theta_b - 2) / (theta_b - 1),
            # u[i,j-2]
            -eps_p * (theta_b - 1) / (theta_b - 2),
            # u[i,j+1]
            eps_jump * n1 * n2 * theta_b * dy / dx
            - (eps_jump * n1**2 + eps_m) * theta_b / (1 + theta_b),
            # u[i-1,j]
            eps_jump * n1 * n2 * (2 * theta_b + 1) * dy / (2 * dx),
            # u[i+1,j]
            -eps_jump * n1 * n2 * dy / (2 * dx),
            # u[i-1,j+1]
            -eps_jump * n1 * n2 * theta_b * dy / dx,
        ]
        u_arr = [u[i,j], u[i,j-1], u[i,j-2], u[i,j+1], u[i-1,j], u[i+1,j], u[i-1,j+1],]
        u_I = (np.dot(N, u_arr) + d)/M
        return u[i-1,j], u[i+1,j], u_I, u[i, j+1], theta_l, theta_r, theta_b, theta_t
    else:
        raise ValueError("Invalid direction for case 1", direction)

def interface_value_case2(direction: int, i: int, j: int, u: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
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
    for i in range(1, nx-1):
        for j in range(1, ny-1):
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
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = interface_value_case0(i, j, u)
                case 1:
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = interface_value_case1(direction, i, j, u)
                case 2:
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = interface_value_case2(direction, i, j, u)
                case _:
                    raise NotImplementedError("More than 2 cuts not implemented yet.")

            dudx[i, j] = (-theta_r**2*u_l + (theta_r**2-theta_l**2)*u[i,j] + theta_l**2*u_r) / (theta_l*theta_r*(theta_l + theta_r) * dx)
            dudy[i, j] = (-theta_t**2*u_b + (theta_t**2-theta_b**2)*u[i,j] + theta_b**2*u_t) / (theta_b*theta_t*(theta_b + theta_t) * dy)    
    return dudx, dudy

def problem1():
    """[beta]=0, a=0, b=0, surface=x-0.52, f=-2pi^2 sin(pi x) sin(pi y)"""
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return x - 0.52

    def normal(x, y):
        return 1.0, 0.0

    def permittivity(x, y):
        return 1.0

    nx, ny = 100, 5
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    print("x=", x)
    print("y=", y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    a = np.zeros((nx, ny))
    b = np.zeros((nx, ny))
    rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
    A = construct_matrix()
    A_dense = A.toarray()
    print("A_dense = \n", A_dense)
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    error = np.max(np.abs(u - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.spy(A)
    # plt.imshow(A.toarray(), interpolation="none", cmap="binary")
    # plt.colorbar()
    plt.title("Sparsity Pattern")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(1, 2, 2)
    # plt.pcolormesh(X, Y, u, shading="auto", vmin=0.0, vmax=1.0)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def problem2():
    """[beta]=0, a=0, b=0, surface=y-0.52, f=-2pi^2 sin(pi x) sin(pi y)"""
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return y - 0.52

    def normal(x, y):
        return 0.0, 1.0

    def permittivity(x, y):
        return 1.0

    nx, ny = 5, 100
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    print("x=", x)
    print("y=", y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    a = np.zeros((nx, ny))
    b = np.zeros((nx, ny))
    rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
    A = construct_matrix()
    A_dense = A.toarray()
    print("A_dense = \n", A_dense)
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    error = np.max(np.abs(u - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.spy(A)
    # plt.imshow(A.toarray(), interpolation="none", cmap="binary")
    # plt.colorbar()
    plt.title("Sparsity Pattern")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def problem3():
    """[beta]=0, a=1, b=0, surface=x-0.52, f=0"""
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return x - 0.52

    def normal(x, y):
        return 1.0, 0.0

    def permittivity(x, y):
        return 1.0

    nx, ny = 100, 5
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    print("x=", x)
    print("y=", y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.zeros((nx, ny))
    u_exact = np.piecewise(X, [X < 0.52, X >= 0.52], [lambda x: x, lambda x: x + 1.0])
    a = np.ones((nx, ny))
    b = np.zeros((nx, ny))
    rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
    A = construct_matrix()
    A_dense = A.toarray()
    print("A_dense = \n", A_dense)
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    error = np.max(np.abs(u - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.spy(A)
    # plt.imshow(A.toarray(), interpolation="none", cmap="binary")
    # plt.colorbar()
    plt.title("Sparsity Pattern")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto")
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(1, 2, 2)
    # plt.pcolormesh(X, Y, u, shading="auto", vmin=0.0, vmax=1.0)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def problem4():
    """[beta]=0, a=1, b=0, surface=y-0.52, f=0"""
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return y - 0.52

    def normal(x, y):
        return 0.0, 1.0

    def permittivity(x, y):
        return 1.0

    nx, ny = 5, 100
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    print("x=", x)
    print("y=", y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.zeros((nx, ny))
    u_exact = np.piecewise(Y, [Y < 0.52, Y >= 0.52], [lambda y: y, lambda y: y + 1.0])
    a = np.ones((nx, ny))
    b = np.zeros((nx, ny))
    rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
    A = construct_matrix()
    A_dense = A.toarray()
    print("A_dense = \n", A_dense)
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    error = np.max(np.abs(u - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.spy(A)
    # plt.imshow(A.toarray(), interpolation="none", cmap="binary")
    # plt.colorbar()
    plt.title("Sparsity Pattern")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto")
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(1, 2, 2)
    # plt.pcolormesh(X, Y, u, shading="auto", vmin=0.0, vmax=1.0)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def problem5():
    """[beta]=0, a=0, b=1, surface=x-0.52, f=0"""
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return x - 0.52

    def normal(x, y):
        return 1.0, 0.0

    def permittivity(x, y):
        return 1.0

    nx, ny = 100, 3
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    print("x=", x)
    print("y=", y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.zeros((nx, ny))
    u_exact = np.piecewise(
        X, [X < 0.52, X >= 0.52], [lambda x: x, lambda x: 2 * (x - 0.52) + 0.52]
    )
    a = np.zeros((nx, ny))
    b = np.ones((nx, ny))
    rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
    A = construct_matrix()
    A_dense = A.toarray()
    print("A_dense = \n", A_dense)
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    error = np.max(np.abs(u - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.spy(A)
    # plt.imshow(A.toarray(), interpolation="none", cmap="binary")
    # plt.colorbar()
    plt.title("Sparsity Pattern")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto")
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.subplot(1, 2, 2)
    # plt.pcolormesh(X, Y, u, shading="auto", vmin=0.0, vmax=1.0)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def problem6():
    """[beta]=0, a=0, b=1, surface=y-0.52, f=0"""
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return y - 0.52

    def normal(x, y):
        return 0.0, 1.0

    def permittivity(x, y):
        return 1.0

    nx, ny = 5, 50
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    print("x=", x)
    print("y=", y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.zeros((nx, ny))
    u_exact = np.piecewise(
        Y, [Y < 0.52, Y >= 0.52], [lambda y: y, lambda y: 2 * (y - 0.52) + 0.52]
    )
    a = np.zeros((nx, ny))
    b = np.ones((nx, ny))
    rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
    A = construct_matrix()
    A_dense = A.toarray()
    print("A_dense = \n", A_dense)
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    error = np.max(np.abs(u - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.spy(A)
    # plt.imshow(A.toarray(), interpolation="none", cmap="binary")
    # plt.colorbar()
    plt.title("Sparsity Pattern")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto")
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def problem7():
    """
    [beta]=-1,
    a(x=0.3)=-exp(-0.09), a(x=0.6)=-exp(-0.36),
    b(x=0.3)=-1.2exp(-0.09), b(x=0.6)=2.4exp(-0.36),
    f=(8x^2-4)exp(-x^2) for 0.3<x<0.6 else 0,
    surface=abs(x-0.45)-0.15
    """
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return np.abs(x - 0.45) - 0.15

    def normal(x, y):
        return (1.0, 0.0) if x >= 0.45 else (-1.0, 0.0)

    def permittivity(x, y):
        return 2.0 if surface(x, y) <= 0 else 1.0

    nx, ny = 50, 3
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    print("x=", x)
    print("y=", y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = np.piecewise(
        X,
        [(X < 0.6) & (X > 0.3)],
        [lambda x: (8 * x**2 - 4) * np.exp(-(x**2)), 0.0],
    )
    u_exact = np.piecewise(
        X,
        [(X < 0.6) & (X > 0.3)],
        [lambda x: np.exp(-(x**2)), 0.0],
    )
    a = np.piecewise(
        X, [X < 0.45, X >= 0.45], [lambda x: -np.exp(-0.09), lambda x: -np.exp(-0.36)]
    )
    b = np.piecewise(
        X,
        [X < 0.45, X >= 0.45],
        [lambda x: -1.2 * np.exp(-0.09), lambda x: 2.4 * np.exp(-0.36)],
    )
    rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
    A = construct_matrix()
    A_dense = A.toarray()
    print("A_dense = \n", A_dense)
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    error = np.max(np.abs(u - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.spy(A)
    # plt.imshow(A.toarray(), interpolation="none", cmap="binary")
    # plt.colorbar()
    plt.title("Sparsity Pattern")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto")
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def problem8():
    """
    [beta]=-1,
    a(y=0.3)=-exp(-0.09), a(y=0.6)=-exp(-0.36),
    b(y=0.3)=-1.2exp(-0.09), b(y=0.6)=2.4exp(-0.36),
    f=(8y^2-4)exp(-y^2) for 0.3<y<0.6 else 0,
    surface=abs(y-0.45)-0.15
    """
    global nx, ny, dx, dy
    global x, y, X, Y
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return np.abs(y - 0.45) - 0.15

    def normal(x, y):
        return (0.0, 1.0) if y >= 0.45 else (0.0, -1.0)

    def permittivity(x, y):
        return 2.0 if surface(x, y) <= 0 else 1.0

    nx, ny = 3, 50
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(dx / 2, 1.0 + dx / 2, dx)
    y = np.arange(dy / 2, 1.0 + dy / 2, dy)
    print("x=", x)
    print("y=", y)
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
    a = np.piecewise(
        Y, [Y < 0.45, Y >= 0.45], [lambda y: -np.exp(-0.09), lambda y: -np.exp(-0.36)]
    )
    b = np.piecewise(
        Y,
        [Y < 0.45, Y >= 0.45],
        [lambda y: -1.2 * np.exp(-0.09), lambda y: 2.4 * np.exp(-0.36)],
    )
    rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
    A = construct_matrix()
    A_dense = A.toarray()
    print("A_dense = \n", A_dense)
    u = spsolve(A, f.flatten())
    u = u.reshape((nx, ny))
    error = np.max(np.abs(u - u_exact))
    print(f"Max error: {error}")
    plt.figure()
    plt.spy(A)
    # plt.imshow(A.toarray(), interpolation="none", cmap="binary")
    # plt.colorbar()
    plt.title("Sparsity Pattern")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, u_exact, shading="auto")
    plt.colorbar(label="u_exact(x,y)")
    plt.title("Exact solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, u, shading="auto")
    plt.colorbar(label="u(x,y)")
    plt.title("Nmerical solution of 2D Poisson equation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


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
    global f, u_exact, a, b
    global rows, cols, vals
    global surface, normal, permittivity

    def surface(x, y):
        return np.abs(y - 0.45) - 0.15

    def normal(x, y):
        return (0.0, 1.0) if y >= 0.45 else (0.0, -1.0)

    def permittivity(x, y):
        return 2.0 if surface(x, y) <= 0 else 1.0

    n_range = 2 ** np.arange(3, 8, dtype=int)
    errors_u = np.zeros(n_range.size)
    errors_du = np.zeros(n_range.size)

    for i, n in enumerate(n_range):
        nx, ny = 3, n
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
        rows, cols, vals = [], [], []  # triplet format for sparse matrix assembly
        A = construct_matrix()
        u = spsolve(A, f.flatten())
        u = u.reshape((nx, ny))
        dudx, dudy = gradient(u)
        errors_u[i] = np.max(np.abs(u - u_exact))
        errors_du[i] = np.max(np.abs(dudx - dudx_exact)[1:-1, 1:-1]) + np.max(np.abs(dudy - dudy_exact)[1:-1, 1:-1])

    plt.figure()
    plt.plot(y, dudy_exact[nx//2,:], label="exact")
    plt.plot(y, dudy[nx//2,:], label="numerical", linestyle="--")
    plt.xlabel("y")
    plt.ylabel("du/dy")
    plt.legend()


    plt.figure()
    plt.subplot(1,2,1)
    plt.loglog(1 / n_range, errors_u, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $u$")
    plt.subplot(1,2,2)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h")
    plt.ylabel("err")
    plt.legend()
    plt.title("Convergence of $\\nabla u$")
    plt.show()


if __name__ == "__main__":
    # problem1()
    # problem2()
    # problem3()
    # problem4()
    # problem5()
    # problem6()
    # problem7()
    # problem8()

    convergence_test1()

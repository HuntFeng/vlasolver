import enum

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class Direction(enum.IntFlag):
    R = 1 << 0  # 0001
    T = 1 << 1  # 0010
    L = 1 << 2  # 0100
    B = 1 << 3  # 1000


def dirsign(direction: int):
    if direction == Direction.R or direction == direction.T:
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
        breakpoint()
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
    cd = _CD[direction]
    if cd.is_x:
        k = i
        points = (
            field[k - 1 : k + 3, j] if cd.sign > 0 else field[k - 2 : k + 2, j][::-1]
        )
    else:
        k = j
        points = (
            field[i, k - 1 : k + 3] if cd.sign > 0 else field[i, k - 2 : k + 2][::-1]
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
                    extra = case3_extra_dir(direction, i, j)
                    if extra is None:
                        coeff_case2(direction, i, j)
                    else:
                        coeff_case3(direction, extra, i, j)
                case 3:
                    coeff_case4(direction, i, j)
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
    eta = surface(x, y)
    theta_r = compute_theta(Direction.R, i, j)
    theta_l = compute_theta(Direction.L, i, j)
    theta_t = compute_theta(Direction.T, i, j)
    theta_b = compute_theta(Direction.B, i, j)

    x_r = x + theta_r * dx
    x_l = x - theta_l * dx
    y_t = y + theta_t * dy
    y_b = y - theta_b * dy
    if direction == Direction.R:
        x_ext = x - dx
        y_ext = y - dy

        eps_m = permittivity(x_r - 0.5 * dx, y)
        eps_p = permittivity(x_r + 0.5 * dx, y)
    elif direction == Direction.T:
        x_ext = x - dx
        y_ext = y - dy

        eps_m = permittivity(x, y_t - 0.5 * dy)
        eps_p = permittivity(x, y_t + 0.5 * dy)
    elif direction == Direction.L:
        x_ext = x + dx
        y_ext = y - dy

        eps_p = permittivity(x_l - 0.5 * dx, y)
        eps_m = permittivity(x_l + 0.5 * dx, y)
    elif direction == Direction.B:
        x_ext = x - dx
        y_ext = y + dy

        eps_p = permittivity(x, y_b - 0.5 * dy)
        eps_m = permittivity(x, y_b + 0.5 * dy)
    else:
        raise ValueError("Invalid direction for coeff_case1", direction)

    if eta > 0:
        eps_m, eps_p = eps_p, eps_m

    P_inv = compute_P_inv(x, y, x_r, x_l, x_ext, y_t, y_b, y_ext)
    a_tau_I = interp(direction, theta, i, j, a_tau)
    a_I = interp(direction, theta, i, j, a)
    b_I = interp(direction, theta, i, j, b)
    n1_I = interp(direction, theta, i, j, n1)
    n2_I = interp(direction, theta, i, j, n2)
    grad_tau = np.array(
        [-2 * x * n2_I, x * n1_I - y * n2_I, 2 * y * n1_I, -n2_I, n1_I, 0.0]
    )
    grad_coeff = -beta_jump * n2_I * grad_tau @ P_inv  # -[beta]ny(grad_P*tau)
    B = beta_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r)) + beta_m * (
        2 * theta_r + 1
    ) / (theta_r * (theta_r + 1))
    C = ...
    a_tau_term = -a_tau_I * beta_p * n2_I
    b_term = b_I * n1_I
    a_term = a_I * beta_p * (3 - 2 * theta_r) / ((1 - theta_r) * (2 - theta_r))

    M = B - grad_coeff[0]
    grad_coeff = np.delete(grad_coeff, 0)
    N = grad_coeff.sum() + ...
    d = a_term - a_tau_term - b_term


dx = dy = 0.1
a_tau = compute_a_tau_field()
n1, n2 = compute_normal_field()

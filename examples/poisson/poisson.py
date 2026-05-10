import enum

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(legacy="1.25")  # no type info when printing

EPS = 0.01

# Shared zero function — all constant-zero matrix entries.
_Z = lambda *_: 0.0


# -- Rational-theta helpers (cubic Hermite interpolation coefficients) --


def _phi(th):
    """_phi(theta) = (3 - 2*th) / ((1-th)*(2-th)), for th in (0,1)."""
    return (3 - 2 * th) / ((1 - th) * (2 - th))


def _psi(th):
    """_psi(theta) = (2*th + 1) / (th*(th+1)), for th in (0,1)."""
    return (2 * th + 1) / (th * (th + 1))


def _phi_coupled(th1, th2):
    """Coupled phi: (2*th1 + th2 - 3) / ((th1-1)*(th1+th2-2))."""
    return (2 * th1 + th2 - 3) / ((th1 - 1) * (th1 + th2 - 2))


def _phi_mirror(th1, th2):
    """Mirror phi: (th1 + 2*th2 - 3) / ((th2-1)*(th1+th2-2))."""
    return (th1 + 2 * th2 - 3) / ((th2 - 1) * (th1 + th2 - 2))


def _couple_off_fwd(th1, th2):
    """Forward off-diagonal coupling: (th1-1) / ((th2-1)*(th1+th2-2))."""
    return (th1 - 1) / ((th2 - 1) * (th1 + th2 - 2))


def _couple_off_rev(th1, th2):
    """Reverse off-diagonal coupling: (th2-1) / ((th1-1)*(th1+th2-2))."""
    return (th2 - 1) / ((th1 - 1) * (th1 + th2 - 2))


def _couple_avg(th1, th2):
    """Average coupling: (th1+th2-2) / ((th1-1)*(th2-1))."""
    return (th1 + th2 - 2) / ((th1 - 1) * (th2 - 1))


# Evaluator dispatch tables
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

np.set_printoptions(legacy="1.25")  # no type info when printing


# ---------------------------------------------------------------------------
# Case 3 / Case 4 evaluators — hardcoded M/N/d formulas live above.
# ---------------------------------------------------------------------------


# Map face name -> row index of the corresponding ghost in the local 3-vector.
_CASE3_SW_INDEX = {
    1: {"R": 0, "T": 2},
    2: {"R": 0, "T": 1},
    3: {"T": 0, "L": 1},
    4: {"T": 0, "L": 1},
    5: {"L": 0, "B": 1},
    6: {"L": 0, "B": 1},
    7: {"B": 0, "R": 1},
    8: {"B": 0, "R": 1},
}
_CASE4_SW_INDEX = {
    1: {"R": 0, "T": 1, "L": 2},
    2: {"R": 0, "T": 1, "B": 2},
    3: {"R": 0, "B": 1, "L": 2},
    4: {"T": 0, "B": 1, "L": 2},
}

# Dispatch tables built once Direction is defined (see _build_dispatch_tables
# below; called right after class Direction).
_CASE3_SUB = {}
_CASE4_SUB = {}


def _build_dispatch_tables():
    _CASE3_SUB.update(
        {
            (Direction.R | Direction.T, Direction.R): 1,
            (Direction.R | Direction.T, Direction.T): 2,
            (Direction.L | Direction.T, Direction.T): 3,
            (Direction.L | Direction.T, Direction.L): 4,
            (Direction.L | Direction.B, Direction.L): 5,
            (Direction.L | Direction.B, Direction.B): 6,
            (Direction.R | Direction.B, Direction.B): 7,
            (Direction.R | Direction.B, Direction.R): 8,
        }
    )
    _CASE4_SUB.update(
        {
            int(Direction.R | Direction.T | Direction.L): 1,
            int(Direction.R | Direction.T | Direction.B): 2,
            int(Direction.R | Direction.B | Direction.L): 3,
            int(Direction.T | Direction.B | Direction.L): 4,
        }
    )


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


_build_dispatch_tables()


# ---------------------------------------------------------------------------
# Direction parameterization — encodes axis / sign to eliminate if/else branches.
# ---------------------------------------------------------------------------


class CutDir:
    """Encodes axis and orientation for a single interface cut direction.

    Each instance captures the mechanical differences between R/L (x-axis,
    forward/backward) and T/B (y-axis, forward/backward) so that downstream
    formulas become a single code path parameterized by these two bits.
    """

    __slots__ = ("face", "axis", "sign")

    def __init__(self, face: int, axis: int, sign: int):
        self.face = face
        self.axis = axis  # 0 → x-axis; 1 → y-axis
        self.sign = sign  # +1 for R/T (forward), -1 for L/B (backward)

    @property
    def is_x(self) -> bool:
        return self.axis == 0

    @property
    def is_y(self) -> bool:
        return self.axis == 1

    @property
    def n_tang(self) -> int:
        """Index of the tangential normal component (0=n1, 1=n2).

        x-axis interfaces (R,L) are vertical → tangent is horizontal → n2.
        y-axis interfaces (T,B) are horizontal → tangent is vertical → n1.
        """
        return 1 if self.is_x else 0

    @property
    def n_norm(self) -> int:
        """Index of the surface-normal component."""
        return 0 if self.is_x else 1

    # -- stencil offsets ------------------------------------------------------
    # The 7 neighbour offsets for the M/d/N system, keyed by semantic role.

    _STENCIL = {
        Direction.R: [(0, 0), (1, 0), (2, 0), (-1, 0), (0, -1), (0, 1), (-1, -1)],
        Direction.T: [(0, 0), (0, 1), (0, 2), (0, -1), (-1, 0), (1, 0), (-1, -1)],
        Direction.L: [(0, 0), (-1, 0), (-2, 0), (1, 0), (0, -1), (0, 1), (1, -1)],
        Direction.B: [(0, 0), (0, -1), (0, -2), (0, 1), (-1, 0), (1, 0), (-1, 1)],
    }

    @property
    def offsets(self):
        """The 7 (di, dj) stencil offsets for the N-vector."""
        return self._STENCIL[self.face]

    # -- theta / eps / bot slot indices (0=L, 1=R, 2=T, 3=B) -----------------

    _SLOT = {Direction.R: 1, Direction.T: 2, Direction.L: 0, Direction.B: 3}

    @property
    def slot(self) -> int:
        """Index into the (L,R,T,B) arrays for the active face."""
        return self._SLOT[self.face]

    def theta_assign(self, theta: float):
        """Return (theta_l, theta_r, theta_t, theta_b) with `theta` in the active slot."""
        t = [1.0, 1.0, 1.0, 1.0]
        t[self.slot] = theta
        return t[0], t[1], t[2], t[3]

    # -- uncut-neighbour correction helpers -----------------------------------

    # Map each face to the (di, dj) of its *neighbour* and which *half-eps*
    # variable corresponds to it.  Used to add eps/theta/bot corrections
    # for the sides that are NOT the active cut.

    _NEIGHBOUR = {
        Direction.R: (1, 0, "eps_r"),
        Direction.L: (-1, 0, "eps_l"),
        Direction.T: (0, 1, "eps_t"),
        Direction.B: (0, -1, "eps_b"),
    }

    @staticmethod
    def uncut_faces(active_face: int):
        """Yield (di, dj, eps_name) for the three faces *other* than `active_face`."""
        for face in (Direction.R, Direction.T, Direction.L, Direction.B):
            if face != active_face:
                yield CutDir._NEIGHBOUR[face]

    # -- beta probe location --------------------------------------------------

    def probe_loc(self, x: float, y: float, theta: float):
        """Return (x_loc, y_loc, axis_str) for _sample_beta_legacy at the interface point."""
        if self.is_x:
            return (x + self.sign * theta * dx, y, "x")
        else:
            return (x, y + self.sign * theta * dy, "y")


# Lookup table Direction int → CutDir instance.
_CD = {
    Direction.R: CutDir(Direction.R, 0, +1),
    Direction.T: CutDir(Direction.T, 1, +1),
    Direction.L: CutDir(Direction.L, 0, -1),
    Direction.B: CutDir(Direction.B, 1, -1),
}

_FACE_NAME = {Direction.R: "R", Direction.T: "T", Direction.L: "L", Direction.B: "B"}


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
    cd = _CD[direction]
    x, y = center(i, j)
    eta = surface(x, y)
    if cd.is_x:
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
        theta = (
            -cd.sign * d_eta - np.sign(eta) * np.sqrt(d_eta**2 - 4 * dd_eta * eta)
        ) / (2 * dd_eta)
    if theta < 1e-6 or theta > 1.0 - 1e-6:
        theta = 1.0
        # TODO: maybe this is not a good idea
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

    D = a_tau_term + b_term - a_term

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
    M_inv_d = D / M

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


def _solve_case2_local(direction: int, i: int, j: int):
    """Build and solve the 2x2 local M / d / N system for a case-2 corner.

    Returns (M_inv_d, M_inv_N, all_offsets, sw_idx, geom) where geom holds
    theta_*, eps_*, bot_* for Shortley-Weller assembly.
    """
    x_, y_ = center(i, j)
    eta = surface(x_, y_)

    cd_x = _CD[direction & (Direction.R | Direction.L)]
    cd_y = _CD[direction & (Direction.T | Direction.B)]

    theta_x = compute_theta(cd_x.face, i, j)
    theta_y = compute_theta(cd_y.face, i, j)

    theta_l = theta_x if cd_x.face == Direction.L else 1.0
    theta_r = theta_x if cd_x.face == Direction.R else 1.0
    theta_t = theta_y if cd_y.face == Direction.T else 1.0
    theta_b = theta_y if cd_y.face == Direction.B else 1.0

    def _interp(cd, theta):
        return (
            interp(cd.face, theta, i, j, n1),
            interp(cd.face, theta, i, j, n2),
            interp(cd.face, theta, i, j, a),
            interp(cd.face, theta, i, j, b),
            interp(cd.face, theta, i, j, a_tau),
        )

    n1_x, n2_x, a_x, b_x, a_tau_x = _interp(cd_x, theta_x)
    n1_y, n2_y, a_y, b_y, a_tau_y = _interp(cd_y, theta_y)

    # Beta sampling at each interface point.
    px, py, _ = cd_x.probe_loc(x_, y_, theta_x)
    bp_x, bm_x, bj_x = _eval_beta_at_iface(px, py, "x" if cd_x.is_x else "y")
    px, py, _ = cd_y.probe_loc(x_, y_, theta_y)
    bp_y, bm_y, bj_y = _eval_beta_at_iface(px, py, "x" if cd_y.is_x else "y")

    def _d_row(cd, theta, n1_v, n2_v, a_v, b_v, a_tau_v, eps_p_d):
        n_t = [n1_v, n2_v][cd.n_tang]
        n_n = [n1_v, n2_v][cd.n_norm]
        d_self = dx if cd.is_x else dy
        return (
            (-1 if cd.is_x else 1) * a_tau_v * eps_p_d * n_t * d_self
            + b_v * n_n * d_self
            + cd.sign * a_v * eps_p_d * _phi(theta)
        )

    eps_p_d_x = bm_x if eta > 0 else bp_x
    eps_p_d_y = bm_y if eta > 0 else bp_y
    d = np.zeros(2)
    d[0] = _d_row(cd_x, theta_x, n1_x, n2_x, a_x, b_x, a_tau_x, eps_p_d_x)
    d[1] = _d_row(cd_y, theta_y, n1_y, n2_y, a_y, b_y, a_tau_y, eps_p_d_y)

    if eta > 0:
        eps_p_x, eps_m_x = -bm_x, -bp_x
        eps_p_y, eps_m_y = -bm_y, -bp_y
    else:
        eps_p_x, eps_m_x = bp_x, bm_x
        eps_p_y, eps_m_y = bp_y, bm_y

    eps_jump_x = bj_x
    eps_jump_y = bj_y

    def _M_diag(cd, theta, n1_v, n2_v, eps_p, eps_m, eps_jump):
        n_t = [n1_v, n2_v][cd.n_tang]
        return -cd.sign * (
            eps_p * _phi(theta) + eps_m * _psi(theta) + eps_jump * n_t**2 * _psi(theta)
        )

    M = np.zeros((2, 2))
    M[0, 0] = _M_diag(cd_x, theta_x, n1_x, n2_x, eps_p_x, eps_m_x, eps_jump_x)
    M[1, 1] = _M_diag(cd_y, theta_y, n1_y, n2_y, eps_p_y, eps_m_y, eps_jump_y)
    M[0, 1] = cd_y.sign * eps_jump_x * n1_x * n2_x * dx / (dy * theta_y * (theta_y + 1))
    M[1, 0] = cd_x.sign * eps_jump_y * n1_y * n2_y * dy / (dx * theta_x * (theta_x + 1))

    # N (2 x 25) — full 5x5 stencil.
    N = np.zeros((2, 25))
    all_offsets = [(ox, oy) for ox in range(-2, 3) for oy in range(-2, 3)]

    def _col(ox, oy):
        return (ox + 2) * 5 + (oy + 2)

    def _fill_N_row(
        r, cd_p, cd_o, theta_p, theta_o, n1_v, n2_v, eps_p, eps_m, eps_jump
    ):
        n_t = [n1_v, n2_v][cd_p.n_tang]
        d_self = dx if cd_p.is_x else dy
        d_other = dy if cd_p.is_x else dx
        off = cd_p.offsets

        # k=0 : self
        N[r, _col(*off[0])] = (
            -cd_p.sign * (eps_jump * n_t**2 + eps_m) * (1 + theta_p) / theta_p
            - cd_o.sign
            * eps_jump
            * n1_v
            * n2_v
            * d_self
            / d_other
            * (theta_o * theta_p + theta_o - 1)
            / theta_o
        )
        # k=1 : forward 1
        N[r, _col(*off[1])] = -cd_p.sign * eps_p * (theta_p - 2) / (theta_p - 1)
        # k=2 : forward 2
        N[r, _col(*off[2])] = cd_p.sign * eps_p * (theta_p - 1) / (theta_p - 2)
        # k=3 : backward
        N[r, _col(*off[3])] = (
            cd_p.sign * (eps_jump * n_t**2 + eps_m) * theta_p / (1 + theta_p)
            + cd_o.sign * eps_jump * n1_v * n2_v * theta_p * d_self / d_other
        )
        # k=4/5 : combined cross-term — k=4 when cd_o.sign > 0, else k=5.
        k_comb = 4 if cd_o.sign > 0 else 5
        N[r, _col(*off[k_comb])] = (
            cd_o.sign
            * eps_jump
            * n1_v
            * n2_v
            * d_self
            / d_other
            * (theta_o / (theta_o + 1) + theta_p)
        )
        # k=6 : corner — offset component along cd_o's axis scaled by cd_o.sign.
        di, dj = off[6]
        if cd_o.is_x:
            di *= cd_o.sign
        else:
            dj *= cd_o.sign
        N[r, _col(di, dj)] = (
            -cd_o.sign * eps_jump * n1_v * n2_v * theta_p * d_self / d_other
        )

    _fill_N_row(
        0, cd_x, cd_y, theta_x, theta_y, n1_x, n2_x, eps_p_x, eps_m_x, eps_jump_x
    )
    _fill_N_row(
        1, cd_y, cd_x, theta_y, theta_x, n1_y, n2_y, eps_p_y, eps_m_y, eps_jump_y
    )

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)

    sw_idx = {_FACE_NAME[cd_x.face]: 0, _FACE_NAME[cd_y.face]: 1}
    geom = {
        "theta_R": theta_r,
        "theta_T": theta_t,
        "theta_L": theta_l,
        "theta_B": theta_b,
        "eps_r": permittivity(x_ + theta_r * dx / 2, y_),
        "eps_l": permittivity(x_ - theta_l * dx / 2, y_),
        "eps_t": permittivity(x_, y_ + theta_t * dy / 2),
        "eps_b": permittivity(x_, y_ - theta_b * dy / 2),
        "bot_x": (theta_r + theta_l) / 2 * dx**2,
        "bot_y": (theta_t + theta_b) / 2 * dy**2,
    }

    return M_inv_d, M_inv_N, all_offsets, sw_idx, geom


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
        -s[d]
        * a_I[d]
        * eps_p[d]
        * (3 - 2 * theta[d])
        / ((1 - theta[d]) * (2 - theta[d]))
        for d in range(2)
    ]

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

    D[:] = [a_tau_term[d] + b_term[d] - a_term[d] for d in range(2)]

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
        N[d, offset(1, 0)] = grad_coeff[d][0] if Direction.R not in dir else 0.0
        N[d, offset(-1, 0)] = grad_coeff[d][1] if Direction.L not in dir else 0.0
        N[d, offset(0, 1)] = grad_coeff[d][2] if Direction.T not in dir else 0.0
        N[d, offset(0, -1)] = grad_coeff[d][3] if Direction.B not in dir else 0.0
        N[d, offset(0, 0)] = grad_coeff[d][4]
        N[d, offset(*offset_ext)] = grad_coeff[d][5]

        # Place [k] along the direction ray at offsets (-1, 0, 1, 2) from center
        dx_dir, dy_dir = _dir_step[dir[d]]
        for k in range(4):
            N[d, offset((k - 1) * dx_dir, (k - 1) * dy_dir)] -= C[d][k]

    M_inv_d = np.linalg.solve(M, D)
    M_inv_N = np.linalg.solve(M, N)

    sub_coeff = [eps[0] / theta[0] / bot_x, eps[1] / theta[1] / bot_y]
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
    # extras = []
    # if direction == (Direction.R | Direction.T):
    #     if surface(x_ + dx, y_) * surface(x_ + 2 * dx, y_) < 0:
    #         extras.append(Direction.R)
    #     if surface(x_, y_ + dy) * surface(x_, y_ + 2 * dy) < 0:
    #         extras.append(Direction.T)
    # elif direction == (Direction.L | Direction.T):
    #     if surface(x_, y_ + dy) * surface(x_, y_ + 2 * dy) < 0:
    #         extras.append(Direction.T)
    #     if surface(x_ - dx, y_) * surface(x_ - 2 * dx, y_) < 0:
    #         extras.append(Direction.L)
    # elif direction == (Direction.L | Direction.B):
    #     if surface(x_ - dx, y_) * surface(x_ - 2 * dx, y_) < 0:
    #         extras.append(Direction.L)
    #     if surface(x_, y_ - dy) * surface(x_, y_ - 2 * dy) < 0:
    #         extras.append(Direction.B)
    # elif direction == (Direction.R | Direction.B):
    #     if surface(x_, y_ - dy) * surface(x_, y_ - 2 * dy) < 0:
    #         extras.append(Direction.B)
    #     if surface(x_ + dx, y_) * surface(x_ + 2 * dx, y_) < 0:
    #         extras.append(Direction.R)
    # for extra in extras.copy():
    #     if (extra == Direction.R and i + 3 >= nx) or (extra == Direction.L and i - 3 < 0) or (extra == Direction.T and j + 3 >= ny) or (extra == Direction.B and j - 3 < 0):
    #         extras.pop()
    # return extras


def _case3_geometry(direction: int, extra: int, i: int, j: int):
    """Compute geometry/normals/jumps at the three interface points.

    Returns a dict with theta_*, theta_*_extra (the lowercase one), normals
    n1_*, n2_* at each of the three interface points, jump conditions
    a, b, a_tau at each, plus eps_p / eps_m / eps_jump (with eta-sign
    handling matching coeff_case2).
    """
    x_, y_ = center(i, j)
    eta = surface(x_, y_)

    theta_R = compute_theta(Direction.R, i, j) if (direction & Direction.R) else 1.0
    theta_T = compute_theta(Direction.T, i, j) if (direction & Direction.T) else 1.0
    theta_L = compute_theta(Direction.L, i, j) if (direction & Direction.L) else 1.0
    theta_B = compute_theta(Direction.B, i, j) if (direction & Direction.B) else 1.0

    # theta for the extra interface — measured from the OUTER end of its
    # segment, matching the derivation file convention:
    #   xr = x_{i+2} - theta_r*dx,  xl = x_{i-2} + theta_l*dx
    #   xt = y_{j+2} - theta_t*dy,  xb = y_{j-2} + theta_b*dy
    if extra == Direction.R:
        theta_extra = compute_theta(Direction.R, i + 1, j)  # within [(i+1,j),(i+2,j)]
    elif extra == Direction.T:
        theta_extra = compute_theta(Direction.T, i, j + 1)
    elif extra == Direction.L:
        theta_extra = compute_theta(Direction.L, i - 1, j)
    elif extra == Direction.B:
        theta_extra = compute_theta(Direction.B, i, j - 1)
    else:
        raise ValueError("bad extra direction", extra)

    # Normals, a_tau, a, b at the three interface points.
    def _at(d, ti, tj, theta):
        return (
            interp(d, theta, ti, tj, n1),
            interp(d, theta, ti, tj, n2),
            interp(d, theta, ti, tj, a),
            interp(d, theta, ti, tj, b),
            interp(d, theta, ti, tj, a_tau),
        )

    geom = {
        "theta_R": theta_R,
        "theta_T": theta_T,
        "theta_L": theta_L,
        "theta_B": theta_B,
        # Paper convention: theta_r/t/l/b is measured from the OUTER end of
        # the outer segment ([x_{i+2}, x_{i+1}] etc.), but `compute_theta`
        # always returns the forward fraction from the inner end. Convert.
        "theta_extra": 1.0 - theta_extra,
        "extra": extra,
    }
    if direction & Direction.R:
        geom["n1_R"], geom["n2_R"], geom["a_R"], geom["b_R"], geom["a_tau_R"] = _at(
            Direction.R, i, j, theta_R
        )
        geom["loc_R"] = (x_ + theta_R * dx, y_)
    if direction & Direction.T:
        geom["n1_T"], geom["n2_T"], geom["a_T"], geom["b_T"], geom["a_tau_T"] = _at(
            Direction.T, i, j, theta_T
        )
        geom["loc_T"] = (x_, y_ + theta_T * dy)
    if direction & Direction.L:
        geom["n1_L"], geom["n2_L"], geom["a_L"], geom["b_L"], geom["a_tau_L"] = _at(
            Direction.L, i, j, theta_L
        )
        geom["loc_L"] = (x_ - theta_L * dx, y_)
    if direction & Direction.B:
        geom["n1_B"], geom["n2_B"], geom["a_B"], geom["b_B"], geom["a_tau_B"] = _at(
            Direction.B, i, j, theta_B
        )
        geom["loc_B"] = (x_, y_ - theta_B * dy)

    # Extra interface point: interpolate from grid points one cell out, and
    # store its absolute (x, y) using the paper's outer-fraction convention.
    if extra == Direction.R:
        geom["n1_x"], geom["n2_x"], geom["a_x"], geom["b_x"], geom["a_tau_x"] = _at(
            Direction.R, i + 1, j, theta_extra
        )
        geom["loc_extra"] = (x_ + 2 * dx - geom["theta_extra"] * dx, y_)
    elif extra == Direction.T:
        geom["n1_x"], geom["n2_x"], geom["a_x"], geom["b_x"], geom["a_tau_x"] = _at(
            Direction.T, i, j + 1, theta_extra
        )
        geom["loc_extra"] = (x_, y_ + 2 * dy - geom["theta_extra"] * dy)
    elif extra == Direction.L:
        geom["n1_x"], geom["n2_x"], geom["a_x"], geom["b_x"], geom["a_tau_x"] = _at(
            Direction.L, i - 1, j, theta_extra
        )
        geom["loc_extra"] = (x_ - 2 * dx + geom["theta_extra"] * dx, y_)
    elif extra == Direction.B:
        geom["n1_x"], geom["n2_x"], geom["a_x"], geom["b_x"], geom["a_tau_x"] = _at(
            Direction.B, i, j - 1, theta_extra
        )
        geom["loc_extra"] = (x_, y_ - 2 * dy + geom["theta_extra"] * dy)

    # Permittivities and eta-sign handling — same convention as coeff_case2.
    if eta > 0:
        _eps_p = permittivity(x_, y_)
        # use the diagonally-opposite-corner point in the +/- region
        if direction == (Direction.R | Direction.T):
            _eps_m = permittivity(x_ + dx, y_ + dy)
        elif direction == (Direction.L | Direction.T):
            _eps_m = permittivity(x_ - dx, y_ + dy)
        elif direction == (Direction.R | Direction.B):
            _eps_m = permittivity(x_ + dx, y_ - dy)
        else:  # L|B
            _eps_m = permittivity(x_ - dx, y_ - dy)
        eps_jump = _eps_p - _eps_m
        eps_p, eps_m = _eps_m, _eps_p
    else:
        if direction == (Direction.R | Direction.T):
            _eps_p = permittivity(x_ + dx, y_ + dy)
        elif direction == (Direction.L | Direction.T):
            _eps_p = permittivity(x_ - dx, y_ + dy)
        elif direction == (Direction.R | Direction.B):
            _eps_p = permittivity(x_ + dx, y_ - dy)
        else:
            _eps_p = permittivity(x_ - dx, y_ - dy)
        _eps_m = permittivity(x_, y_)
        eps_jump = _eps_p - _eps_m
        eps_p, eps_m = _eps_p, _eps_m

    geom["eps"] = (_eps_p, _eps_m, eps_jump, eps_p, eps_m)
    geom["eta"] = eta
    return geom


def coeff_case3(direction: int, extra: int, i: int, j: int) -> None:
    x, y = center(i, j)
    eta = surface(x, y)
    s_eta = np.sign(eta)
    row_idx = index(i, j)

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

    M = np.zeros((3, 3))
    N = np.zeros((3, 49))  # 7x7 stencil around (i,j)
    D = np.zeros(3)

    D[:] = [a_tau_term[d] + b_term[d] - a_term[d] for d in range(3)]

    _grad_idx = {Direction.R: 0, Direction.L: 1, Direction.T: 2, Direction.B: 3}
    M = B
    M[0, 0] -= grad_coeff[0][_grad_idx[dir[0]]]
    M[0, 1] -= grad_coeff[0][_grad_idx[dir[1]]]
    M[1, 0] -= grad_coeff[1][_grad_idx[dir[0]]]
    M[1, 1] -= grad_coeff[1][_grad_idx[dir[1]]]
    M[2, 0] -= grad_coeff[2][_grad_idx[dir[0]]]
    M[2, 1] -= grad_coeff[2][_grad_idx[dir[1]]]

    _dir_step = {
        Direction.R: (1, 0),
        Direction.T: (0, 1),
        Direction.L: (-1, 0),
        Direction.B: (0, -1),
    }
    for d in range(3):
        N[d, offset(1, 0)] = grad_coeff[d][0] if dir[0] != Direction.R else 0.0
        N[d, offset(-1, 0)] = grad_coeff[d][1] if dir[0] != Direction.L else 0.0
        N[d, offset(0, 1)] = grad_coeff[d][2] if dir[1] != Direction.T else 0.0
        N[d, offset(0, -1)] = grad_coeff[d][3] if dir[1] != Direction.B else 0.0
        N[d, offset(0, 0)] = grad_coeff[d][4]
        N[d, offset(*offset_ext)] = grad_coeff[d][5]

        # Place [k] along the direction ray at offsets (-1, 0, 1, 2) from center
        dx_dir, dy_dir = _dir_step[dir[d]]
        for k in range(5):
            N[d, offset((k - 1) * dx_dir, (k - 1) * dy_dir)] -= C[d][k]

    M_inv_d = np.linalg.solve(M, D)
    M_inv_N = np.linalg.solve(M, N)

    sub_coeff = [eps[0] / theta[0] / bot_x, eps[1] / theta[1] / bot_y]
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

    for offset_x in range(-3, 4):
        for offset_y in range(-3, 4):
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


def _case4_geometry(direction: int, i: int, j: int):
    """Compute geometry/normals/jumps at the three corner interfaces.

    `direction` is a 3-bit value (any 3 of R,T,L,B). The uncut side gets
    theta = 1.0. Returns the same kind of dict as `_case3_geometry` minus the
    'extra' fields.
    """
    x_, y_ = center(i, j)
    eta = surface(x_, y_)

    theta_R = compute_theta(Direction.R, i, j) if (direction & Direction.R) else 1.0
    theta_T = compute_theta(Direction.T, i, j) if (direction & Direction.T) else 1.0
    theta_L = compute_theta(Direction.L, i, j) if (direction & Direction.L) else 1.0
    theta_B = compute_theta(Direction.B, i, j) if (direction & Direction.B) else 1.0

    def _at(d, theta):
        return (
            interp(d, theta, i, j, n1),
            interp(d, theta, i, j, n2),
            interp(d, theta, i, j, a),
            interp(d, theta, i, j, b),
            interp(d, theta, i, j, a_tau),
        )

    geom = {
        "theta_R": theta_R,
        "theta_T": theta_T,
        "theta_L": theta_L,
        "theta_B": theta_B,
    }
    if direction & Direction.R:
        geom["n1_R"], geom["n2_R"], geom["a_R"], geom["b_R"], geom["a_tau_R"] = _at(
            Direction.R, theta_R
        )
        geom["loc_R"] = (x_ + theta_R * dx, y_)
    if direction & Direction.T:
        geom["n1_T"], geom["n2_T"], geom["a_T"], geom["b_T"], geom["a_tau_T"] = _at(
            Direction.T, theta_T
        )
        geom["loc_T"] = (x_, y_ + theta_T * dy)
    if direction & Direction.L:
        geom["n1_L"], geom["n2_L"], geom["a_L"], geom["b_L"], geom["a_tau_L"] = _at(
            Direction.L, theta_L
        )
        geom["loc_L"] = (x_ - theta_L * dx, y_)
    if direction & Direction.B:
        geom["n1_B"], geom["n2_B"], geom["a_B"], geom["b_B"], geom["a_tau_B"] = _at(
            Direction.B, theta_B
        )
        geom["loc_B"] = (x_, y_ - theta_B * dy)

    # Permittivity sign convention follows coeff_case2 / coeff_case3:
    # the "diagonal" point used for eps_p / eps_m is the corner that sits
    # in the opposite region from (i,j). For case 4 we pick a Ω^+ corner
    # adjacent to two of the three cut sides.
    if direction == (Direction.R | Direction.T | Direction.L):
        diag = (x_ + dx, y_ + dy)  # any +x or +y neighbour is in Omega^+
    elif direction == (Direction.R | Direction.T | Direction.B):
        diag = (x_ + dx, y_ + dy)
    elif direction == (Direction.R | Direction.B | Direction.L):
        diag = (x_ + dx, y_ - dy)
    elif direction == (Direction.T | Direction.B | Direction.L):
        diag = (x_ - dx, y_ + dy)
    else:
        raise ValueError("Invalid direction for case 4", direction)

    if eta > 0:
        _eps_p = permittivity(x_, y_)
        _eps_m = permittivity(*diag)
        eps_jump = _eps_p - _eps_m
        eps_p, eps_m = _eps_m, _eps_p
    else:
        _eps_p = permittivity(*diag)
        _eps_m = permittivity(x_, y_)
        eps_jump = _eps_p - _eps_m
        eps_p, eps_m = _eps_p, _eps_m

    geom["eps"] = (_eps_p, _eps_m, eps_jump, eps_p, eps_m)
    geom["eta"] = eta
    return geom


def coeff_case4(direction: int, i: int, j: int) -> None:
    """coeff of u_ij and its neighbors for a case 4 cell (3 cut interfaces)."""
    x, y = center(i, j)
    eta = surface(x, y)
    row_idx = index(i, j)

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

    B = [
        s_eta
        * s[d]
        * (
            eps_p[d] * (3 - 2 * theta[d]) / ((1 - theta[d]) * (2 - theta[d]))
            + eps_m[d] * (2 * theta[d] + 1) / (theta[d] * (theta[d] + 1))
        )
        for d in range(3)
    ]
    C = [
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
        for d in range(3)
    ]
    a_term = [
        -s[d]
        * a_I[d]
        * eps_p[d]
        * (3 - 2 * theta[d])
        / ((1 - theta[d]) * (2 - theta[d]))
        for d in range(3)
    ]

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

    D = np.array([a_tau_term[d] + b_term[d] - a_term[d] for d in range(3)])

    _grad_idx = {Direction.R: 0, Direction.L: 1, Direction.T: 2, Direction.B: 3}
    M = np.zeros((3, 3))
    for d in range(3):
        M[d, d] = B[d]
        for e in range(3):
            M[d, e] -= grad_coeff[d][_grad_idx[dir[e]]]

    N = np.zeros((3, 25))
    _dir_step = {
        Direction.R: (1, 0),
        Direction.T: (0, 1),
        Direction.L: (-1, 0),
        Direction.B: (0, -1),
    }
    for d in range(3):
        N[d, offset(1, 0)] = grad_coeff[d][0] if Direction.R not in dir else 0.0
        N[d, offset(-1, 0)] = grad_coeff[d][1] if Direction.L not in dir else 0.0
        N[d, offset(0, 1)] = grad_coeff[d][2] if Direction.T not in dir else 0.0
        N[d, offset(0, -1)] = grad_coeff[d][3] if Direction.B not in dir else 0.0
        N[d, offset(0, 0)] = grad_coeff[d][4]
        N[d, offset(*offset_ext)] = grad_coeff[d][5]

        dx_dir, dy_dir = _dir_step[dir[d]]
        for k in range(4):
            N[d, offset((k - 1) * dx_dir, (k - 1) * dy_dir)] -= C[d][k]

    M_inv_d = np.linalg.solve(M, D)
    M_inv_N = np.linalg.solve(M, N)

    sub_coeff = [eps[d] / theta[d] / (bot_x if is_x[d] else bot_y) for d in range(3)]
    f[i, j] -= sum(M_inv_d[d] * sub_coeff[d] for d in range(3))

    add_terms = {}
    if Direction.R not in dir:
        add_terms[(1, 0)] = eps_r / theta_r / bot_x
    if Direction.L not in dir:
        add_terms[(-1, 0)] = eps_l / theta_l / bot_x
    if Direction.T not in dir:
        add_terms[(0, 1)] = eps_t / theta_t / bot_y
    if Direction.B not in dir:
        add_terms[(0, -1)] = eps_b / theta_b / bot_y

    for offset_x in range(-2, 3):
        for offset_y in range(-2, 3):
            value = sum(
                M_inv_N[d, offset(offset_x, offset_y)] * sub_coeff[d] for d in range(3)
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


# ---------------------------------------------------------------------------
# Local 3x3 solve and Shortley-Weller assembly (shared by case 3 and case 4)
# ---------------------------------------------------------------------------


def _row_geom_data(geom, iface):
    """Return (n1, n2, a, b, a_tau) for one row's interface label."""
    if iface == "extra":
        return (geom["n1_x"], geom["n2_x"], geom["a_x"], geom["b_x"], geom["a_tau_x"])
    return (
        geom[f"n1_{iface}"],
        geom[f"n2_{iface}"],
        geom[f"a_{iface}"],
        geom[f"b_{iface}"],
        geom[f"a_tau_{iface}"],
    )


def _sample_beta_legacy(x_loc, y_loc, axis, eta):
    """Per-interface beta sampling for case 1 / case 2.

    Returns (_eps_p, _eps_m, eps_jump, eps_p, eps_m) following the same
    convention used inline by the original case 1 / case 2 code:
        _eps_p / _eps_m : beta on Omega+ / Omega- side of the interface.
        eps_jump        : _eps_p - _eps_m.
        eps_p / eps_m   : the d-expression names — for eta > 0 the original
                          code applies a "swap" so that eps_p ends up holding
                          the value of beta on the side opposite the home
                          grid point. We replicate that here.
    """
    _eps_p, _eps_m, eps_jump = _eval_beta_at_iface(x_loc, y_loc, axis)
    if eta > 0:
        eps_p, eps_m = _eps_m, _eps_p
    else:
        eps_p, eps_m = _eps_p, _eps_m
    return _eps_p, _eps_m, eps_jump, eps_p, eps_m


def _eval_beta_at_iface(x_loc, y_loc, axis):
    """Sample (beta_p, beta_m, beta_jump) at an interface point.

    `axis` is the Cartesian direction the interface segment lies on
    ('x' for xR/xL/xr/xl, 'y' for xT/xB/xt/xb). We probe `permittivity`
    a tiny distance to either side along that axis; the probe whose
    level-set value is positive is Omega^+, the other is Omega^-.

    For piecewise-constant beta this reduces to the same value as the
    coarser sampling we used to do; for variable beta it removes the
    O(h) sampling error introduced by evaluating beta one cell away
    from the interface.
    """
    if axis == "x":
        eps_off = 0.01 * dx
        p_plus = (x_loc + eps_off, y_loc)
        p_minus = (x_loc - eps_off, y_loc)
    else:
        eps_off = 0.01 * dy
        p_plus = (x_loc, y_loc + eps_off)
        p_minus = (x_loc, y_loc - eps_off)
    # Decide which probe is Omega^+ by the sign of the level set there.
    if surface(*p_plus) > 0:
        bp = permittivity(*p_plus)
        bm = permittivity(*p_minus)
    else:
        bp = permittivity(*p_minus)
        bm = permittivity(*p_plus)
    return bp, bm, bp - bm


_AXIS_OF_FACE = {"R": "x", "L": "x", "T": "y", "B": "y"}


def _iface_axis(label, extra):
    """Cartesian axis of a row's interface segment ('x' or 'y')."""
    if label == "extra":
        if extra in (Direction.R, Direction.L):
            return "x"
        return "y"  # extra is Direction.T or Direction.B
    return _AXIS_OF_FACE[label]


def _row_betas(geom, row_ifaces, extra=None):
    """Build a list of per-row (bp, bm, bj) by sampling beta at each
    row's own interface point."""
    out = []
    for iface in row_ifaces:
        if iface == "extra":
            x_loc, y_loc = geom["loc_extra"]
        else:
            x_loc, y_loc = geom[f"loc_{iface}"]
        axis = _iface_axis(iface, extra)
        out.append(_eval_beta_at_iface(x_loc, y_loc, axis))
    return out


def _solve_case3_local(sub_case, eta, direction, extra, geom):
    eta_sign = -1 if eta < 0 else +1
    func = CASE3_FUNCS[(sub_case, eta_sign)]

    theta_inputs = {
        "R": geom["theta_R"],
        "T": geom["theta_T"],
        "L": geom["theta_L"],
        "B": geom["theta_B"],
        "r": 1.0,
        "t": 1.0,
        "l": 1.0,
        "b": 1.0,
    }
    if extra == Direction.R:
        theta_inputs["r"] = geom["theta_extra"]
    elif extra == Direction.T:
        theta_inputs["t"] = geom["theta_extra"]
    elif extra == Direction.L:
        theta_inputs["l"] = geom["theta_extra"]
    elif extra == Direction.B:
        theta_inputs["b"] = geom["theta_extra"]

    betas_per_row = _row_betas(
        geom, CASE3_ROW_IFACES[(sub_case, eta_sign)], extra=extra
    )
    return func(geom, theta_inputs, betas_per_row)


def _solve_case4_local(sub_case, eta, direction, geom):
    eta_sign = -1 if eta < 0 else +1
    func = CASE4_FUNCS[(sub_case, eta_sign)]

    theta_inputs = {
        "R": geom["theta_R"],
        "T": geom["theta_T"],
        "L": geom["theta_L"],
        "B": geom["theta_B"],
        "r": 1.0,
        "t": 1.0,
        "l": 1.0,
        "b": 1.0,
    }
    betas_per_row = _row_betas(geom, CASE4_ROW_IFACES[(sub_case, eta_sign)])
    return func(geom, theta_inputs, betas_per_row)


# Generated case functions


def _case3_sc1_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 1, eta < 0.  Row 1 is extra (R,T corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "extra")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "T")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, -1),
        (-1, +0),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, +0),
        (+2, -1),
        (+2, +0),
        (+2, +1),
        (+3, -1),
        (+3, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bp_0 * _phi_coupled(R, r) - _psi(R) * (bj_0 * n2_0**2 + bm_0)
    M[0, 1] = bp_0 * _couple_off_fwd(R, r)
    M[0, 2] = bj_0 * dx * n1_0 * n2_0 / (dy * T * (T + 1))
    M[1, 0] = -bp_1 * _couple_off_rev(R, r)
    M[1, 1] = -bp_1 * _phi_mirror(R, r) + _psi(r) * (bj_1 * n2_1**2 + bm_1)
    M[2, 0] = bj_2 * dy * n1_2 * n2_2 / (dx * R * (R + 1))
    M[2, 2] = bp_2 * -_phi(T) - _psi(T) * (bj_2 * n1_2**2 + bm_2)

    d = np.zeros(3)
    d[0] = (
        -a_tau_0 * bp_0 * dx * n2_0 - a_0 * bp_0 * _couple_avg(R, r) + b_0 * dx * n1_0
    )
    d[1] = (
        -a_tau_1 * bp_1 * dx * n2_1 + a_1 * bp_1 * _couple_avg(R, r) + b_1 * dx * n1_1
    )
    d[2] = a_tau_2 * bp_2 * dy * n1_2 - a_2 * bp_2 * -_phi(T) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 0] = -bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, -1)
    N[0, 1] = (
        R
        * (
            bj_0 * dx * n1_0 * n2_0 * R
            + bj_0 * dx * n1_0 * n2_0
            + bj_0 * dy * n2_0**2
            + bm_0 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[0, 2] = bj_0 * dx * n1_0 * n2_0 * (R * T + R + T) / (dy * (T + 1))  # (+0, -1)
    N[0, 3] = -(
        bj_0 * dx * n1_0 * n2_0 * R**2 * T
        + bj_0 * dx * n1_0 * n2_0 * R * T
        - bj_0 * dx * n1_0 * n2_0 * R
        + bj_0 * dy * n2_0**2 * R * T
        + bj_0 * dy * n2_0**2 * T
        + bm_0 * dy * R * T
        + bm_0 * dy * T
    ) / (dy * R * T)  # (+0, +0)
    N[0, 6] = bp_0 * _couple_avg(R, r)  # (+1, +0)
    N[1, 6] = -bp_1 * _couple_avg(R, r)  # (+1, +0)
    N[1, 7] = (1 / 2) * bj_1 * dx * n1_1 * n2_1 * (2 * r + 1) / dy  # (+2, -1)
    N[1, 8] = -(
        bj_1 * dx * n1_1 * n2_1 * r**2
        - bj_1 * dy * n2_1**2 * r
        - bj_1 * dy * n2_1**2
        - bm_1 * dy * r
        - bm_1 * dy
    ) / (dy * r)  # (+2, +0)
    N[1, 9] = -1 / 2 * bj_1 * dx * n1_1 * n2_1 / dy  # (+2, +1)
    N[1, 10] = -bj_1 * dx * n1_1 * n2_1 * r / dy  # (+3, -1)
    N[1, 11] = (
        r
        * (
            bj_1 * dx * n1_1 * n2_1 * r
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            - bm_1 * dy
        )
        / (dy * (r + 1))
    )  # (+3, +0)
    N[2, 0] = -bj_2 * dy * n1_2 * n2_2 * T / dx  # (-1, -1)
    N[2, 1] = bj_2 * dy * n1_2 * n2_2 * (R * T + R + T) / (dx * (R + 1))  # (-1, +0)
    N[2, 2] = (
        T
        * (
            bj_2 * dx * n1_2**2
            + bj_2 * dy * n1_2 * n2_2 * T
            + bj_2 * dy * n1_2 * n2_2
            + bm_2 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[2, 3] = -(
        bj_2 * dx * n1_2**2 * R * T
        + bj_2 * dx * n1_2**2 * R
        + bj_2 * dy * n1_2 * n2_2 * R * T**2
        + bj_2 * dy * n1_2 * n2_2 * R * T
        - bj_2 * dy * n1_2 * n2_2 * T
        + bm_2 * dx * R * T
        + bm_2 * dx * R
    ) / (dx * R * T)  # (+0, +0)
    N[2, 4] = -bp_2 * (T - 2) / (T - 1)  # (+0, +1)
    N[2, 5] = bp_2 * (T - 1) / (T - 2)  # (+0, +2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc1_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 1, eta > 0.  Row 1 is extra (R,T corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "extra")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "T")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, -1),
        (-1, +0),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, +0),
        (+2, -1),
        (+2, +0),
        (+2, +1),
        (+3, -1),
        (+3, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bm_0 * _phi_coupled(R, r) - _psi(R) * (bj_0 * n2_0**2 - bp_0)
    M[0, 1] = -bm_0 * _couple_off_fwd(R, r)
    M[0, 2] = bj_0 * dx * n1_0 * n2_0 / (dy * T * (T + 1))
    M[1, 0] = bm_1 * _couple_off_rev(R, r)
    M[1, 1] = bm_1 * _phi_mirror(R, r) + _psi(r) * (bj_1 * n2_1**2 - bp_1)
    M[2, 0] = bj_2 * dy * n1_2 * n2_2 / (dx * R * (R + 1))
    M[2, 2] = -bm_2 * -_phi(T) - _psi(T) * (bj_2 * n1_2**2 - bp_2)

    d = np.zeros(3)
    d[0] = (
        -a_tau_0 * bm_0 * dx * n2_0 - a_0 * bm_0 * _couple_avg(R, r) + b_0 * dx * n1_0
    )
    d[1] = (
        -a_tau_1 * bm_1 * dx * n2_1 + a_1 * bm_1 * _couple_avg(R, r) + b_1 * dx * n1_1
    )
    d[2] = a_tau_2 * bm_2 * dy * n1_2 - a_2 * bm_2 * -_phi(T) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 0] = -bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, -1)
    N[0, 1] = (
        R
        * (
            bj_0 * dx * n1_0 * n2_0 * R
            + bj_0 * dx * n1_0 * n2_0
            + bj_0 * dy * n2_0**2
            - bp_0 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[0, 2] = bj_0 * dx * n1_0 * n2_0 * (R * T + R + T) / (dy * (T + 1))  # (+0, -1)
    N[0, 3] = -(
        bj_0 * dx * n1_0 * n2_0 * R**2 * T
        + bj_0 * dx * n1_0 * n2_0 * R * T
        - bj_0 * dx * n1_0 * n2_0 * R
        + bj_0 * dy * n2_0**2 * R * T
        + bj_0 * dy * n2_0**2 * T
        - bp_0 * dy * R * T
        - bp_0 * dy * T
    ) / (dy * R * T)  # (+0, +0)
    N[0, 6] = -bm_0 * _couple_avg(R, r)  # (+1, +0)
    N[1, 6] = bm_1 * _couple_avg(R, r)  # (+1, +0)
    N[1, 7] = (1 / 2) * bj_1 * dx * n1_1 * n2_1 * (2 * r + 1) / dy  # (+2, -1)
    N[1, 8] = -(
        bj_1 * dx * n1_1 * n2_1 * r**2
        - bj_1 * dy * n2_1**2 * r
        - bj_1 * dy * n2_1**2
        + bp_1 * dy * r
        + bp_1 * dy
    ) / (dy * r)  # (+2, +0)
    N[1, 9] = -1 / 2 * bj_1 * dx * n1_1 * n2_1 / dy  # (+2, +1)
    N[1, 10] = -bj_1 * dx * n1_1 * n2_1 * r / dy  # (+3, -1)
    N[1, 11] = (
        r
        * (
            bj_1 * dx * n1_1 * n2_1 * r
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            + bp_1 * dy
        )
        / (dy * (r + 1))
    )  # (+3, +0)
    N[2, 0] = -bj_2 * dy * n1_2 * n2_2 * T / dx  # (-1, -1)
    N[2, 1] = bj_2 * dy * n1_2 * n2_2 * (R * T + R + T) / (dx * (R + 1))  # (-1, +0)
    N[2, 2] = (
        T
        * (
            bj_2 * dx * n1_2**2
            + bj_2 * dy * n1_2 * n2_2 * T
            + bj_2 * dy * n1_2 * n2_2
            - bp_2 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[2, 3] = -(
        bj_2 * dx * n1_2**2 * R * T
        + bj_2 * dx * n1_2**2 * R
        + bj_2 * dy * n1_2 * n2_2 * R * T**2
        + bj_2 * dy * n1_2 * n2_2 * R * T
        - bj_2 * dy * n1_2 * n2_2 * T
        - bp_2 * dx * R * T
        - bp_2 * dx * R
    ) / (dx * R * T)  # (+0, +0)
    N[2, 4] = bm_2 * (T - 2) / (T - 1)  # (+0, +1)
    N[2, 5] = -bm_2 * (T - 1) / (T - 2)  # (+0, +2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc2_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 2, eta < 0.  Row 2 is extra (R,T corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "T")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, -1),
        (-1, +0),
        (-1, +2),
        (-1, +3),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+0, +3),
        (+1, +0),
        (+1, +2),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bp_0 * -_phi(R) - _psi(R) * (bj_0 * n2_0**2 + bm_0)
    M[0, 1] = bj_0 * dx * n1_0 * n2_0 / (dy * T * (T + 1))
    M[1, 0] = bj_1 * dy * n1_1 * n2_1 / (dx * R * (R + 1))
    M[1, 1] = bp_1 * _phi_coupled(T, t) - _psi(T) * (bj_1 * n1_1**2 + bm_1)
    M[1, 2] = bp_1 * _couple_off_fwd(T, t)
    M[2, 2] = -bp_2 * _couple_off_rev(T, t)

    d = np.zeros(3)
    d[0] = -a_tau_0 * bp_0 * dx * n2_0 - a_0 * bp_0 * -_phi(R) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bp_1 * dy * n1_1 - a_1 * bp_1 * _couple_avg(T, t) + b_1 * dy * n2_1
    d[2] = a_tau_2 * bp_2 * dy * n1_2 + a_2 * bp_2 * _couple_avg(T, t) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 0] = -bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, -1)
    N[0, 1] = (
        R
        * (
            bj_0 * dx * n1_0 * n2_0 * R
            + bj_0 * dx * n1_0 * n2_0
            + bj_0 * dy * n2_0**2
            + bm_0 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[0, 4] = bj_0 * dx * n1_0 * n2_0 * (R * T + R + T) / (dy * (T + 1))  # (+0, -1)
    N[0, 5] = -(
        bj_0 * dx * n1_0 * n2_0 * R**2 * T
        + bj_0 * dx * n1_0 * n2_0 * R * T
        - bj_0 * dx * n1_0 * n2_0 * R
        + bj_0 * dy * n2_0**2 * R * T
        + bj_0 * dy * n2_0**2 * T
        + bm_0 * dy * R * T
        + bm_0 * dy * T
    ) / (dy * R * T)  # (+0, +0)
    N[0, 9] = -bp_0 * (R - 2) / (R - 1)  # (+1, +0)
    N[0, 11] = bp_0 * (R - 1) / (R - 2)  # (+2, +0)
    N[1, 0] = -bj_1 * dy * n1_1 * n2_1 * T / dx  # (-1, -1)
    N[1, 1] = bj_1 * dy * n1_1 * n2_1 * (R * T + R + T) / (dx * (R + 1))  # (-1, +0)
    N[1, 4] = (
        T
        * (
            bj_1 * dx * n1_1**2
            + bj_1 * dy * n1_1 * n2_1 * T
            + bj_1 * dy * n1_1 * n2_1
            + bm_1 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[1, 5] = -(
        bj_1 * dx * n1_1**2 * R * T
        + bj_1 * dx * n1_1**2 * R
        + bj_1 * dy * n1_1 * n2_1 * R * T**2
        + bj_1 * dy * n1_1 * n2_1 * R * T
        - bj_1 * dy * n1_1 * n2_1 * T
        + bm_1 * dx * R * T
        + bm_1 * dx * R
    ) / (dx * R * T)  # (+0, +0)
    N[1, 6] = bp_1 * _couple_avg(T, t)  # (+0, +1)
    N[2, 2] = (1 / 2) * bj_2 * dy * n1_2 * n2_2 * (2 * t + 1) / dx  # (-1, +2)
    N[2, 3] = -bj_2 * dy * n1_2 * n2_2 * t / dx  # (-1, +3)
    N[2, 6] = -bp_2 * _couple_avg(T, t)  # (+0, +1)
    N[2, 7] = (
        bj_2 * dx * n1_2**2 * t
        + bj_2 * dx * n1_2**2
        - bj_2 * dy * n1_2 * n2_2 * t**2
        + bm_2 * dx * t
        + bm_2 * dx
    ) / (dx * t)  # (+0, +2)
    N[2, 8] = (
        -t
        * (
            bj_2 * dx * n1_2**2
            - bj_2 * dy * n1_2 * n2_2 * t
            - bj_2 * dy * n1_2 * n2_2
            + bm_2 * dx
        )
        / (dx * (t + 1))
    )  # (+0, +3)
    N[2, 10] = -1 / 2 * bj_2 * dy * n1_2 * n2_2 / dx  # (+1, +2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc2_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 2, eta > 0.  Row 2 is extra (R,T corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "T")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, -1),
        (-1, +0),
        (-1, +2),
        (-1, +3),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+0, +3),
        (+1, +0),
        (+1, +2),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bm_0 * -_phi(R) - _psi(R) * (bj_0 * n2_0**2 - bp_0)
    M[0, 1] = bj_0 * dx * n1_0 * n2_0 / (dy * T * (T + 1))
    M[1, 0] = bj_1 * dy * n1_1 * n2_1 / (dx * R * (R + 1))
    M[1, 1] = -bm_1 * _phi_coupled(T, t) - _psi(T) * (bj_1 * n1_1**2 - bp_1)
    M[1, 2] = -bm_1 * _couple_off_fwd(T, t)
    M[2, 2] = bm_2 * _couple_off_rev(T, t)

    d = np.zeros(3)
    d[0] = -a_tau_0 * bm_0 * dx * n2_0 - a_0 * bm_0 * -_phi(R) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bm_1 * dy * n1_1 - a_1 * bm_1 * _couple_avg(T, t) + b_1 * dy * n2_1
    d[2] = a_tau_2 * bm_2 * dy * n1_2 + a_2 * bm_2 * _couple_avg(T, t) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 0] = -bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, -1)
    N[0, 1] = (
        R
        * (
            bj_0 * dx * n1_0 * n2_0 * R
            + bj_0 * dx * n1_0 * n2_0
            + bj_0 * dy * n2_0**2
            - bp_0 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[0, 4] = bj_0 * dx * n1_0 * n2_0 * (R * T + R + T) / (dy * (T + 1))  # (+0, -1)
    N[0, 5] = -(
        bj_0 * dx * n1_0 * n2_0 * R**2 * T
        + bj_0 * dx * n1_0 * n2_0 * R * T
        - bj_0 * dx * n1_0 * n2_0 * R
        + bj_0 * dy * n2_0**2 * R * T
        + bj_0 * dy * n2_0**2 * T
        - bp_0 * dy * R * T
        - bp_0 * dy * T
    ) / (dy * R * T)  # (+0, +0)
    N[0, 9] = bm_0 * (R - 2) / (R - 1)  # (+1, +0)
    N[0, 11] = -bm_0 * (R - 1) / (R - 2)  # (+2, +0)
    N[1, 0] = -bj_1 * dy * n1_1 * n2_1 * T / dx  # (-1, -1)
    N[1, 1] = bj_1 * dy * n1_1 * n2_1 * (R * T + R + T) / (dx * (R + 1))  # (-1, +0)
    N[1, 4] = (
        T
        * (
            bj_1 * dx * n1_1**2
            + bj_1 * dy * n1_1 * n2_1 * T
            + bj_1 * dy * n1_1 * n2_1
            - bp_1 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[1, 5] = -(
        bj_1 * dx * n1_1**2 * R * T
        + bj_1 * dx * n1_1**2 * R
        + bj_1 * dy * n1_1 * n2_1 * R * T**2
        + bj_1 * dy * n1_1 * n2_1 * R * T
        - bj_1 * dy * n1_1 * n2_1 * T
        - bp_1 * dx * R * T
        - bp_1 * dx * R
    ) / (dx * R * T)  # (+0, +0)
    N[1, 6] = -bm_1 * _couple_avg(T, t)  # (+0, +1)
    N[2, 2] = (1 / 2) * bj_2 * dy * n1_2 * n2_2 * (2 * t + 1) / dx  # (-1, +2)
    N[2, 3] = -bj_2 * dy * n1_2 * n2_2 * t / dx  # (-1, +3)
    N[2, 6] = bm_2 * _couple_avg(T, t)  # (+0, +1)
    N[2, 7] = (
        bj_2 * dx * n1_2**2 * t
        + bj_2 * dx * n1_2**2
        - bj_2 * dy * n1_2 * n2_2 * t**2
        - bp_2 * dx * t
        - bp_2 * dx
    ) / (dx * t)  # (+0, +2)
    N[2, 8] = (
        -t
        * (
            bj_2 * dx * n1_2**2
            - bj_2 * dy * n1_2 * n2_2 * t
            - bj_2 * dy * n1_2 * n2_2
            - bp_2 * dx
        )
        / (dx * (t + 1))
    )  # (+0, +3)
    N[2, 10] = -1 / 2 * bj_2 * dy * n1_2 * n2_2 / dx  # (+1, +2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc3_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 3, eta < 0.  Row 2 is extra (T,L corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "T")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "L")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, +0),
        (-1, +2),
        (-1, +3),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+0, +3),
        (+1, -1),
        (+1, +0),
        (+1, +2),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bp_0 * _phi_coupled(T, t) - _psi(T) * (bj_0 * n1_0**2 + bm_0)
    M[0, 1] = -bj_0 * dy * n1_0 * n2_0 / (dx * L * (L + 1))
    M[0, 2] = bp_0 * _couple_off_fwd(T, t)
    M[1, 0] = bj_1 * dx * n1_1 * n2_1 / (dy * T * (T + 1))
    M[1, 1] = -bp_1 * -_phi(L) + _psi(L) * (bj_1 * n2_1**2 + bm_1)
    M[2, 0] = -bp_2 * _couple_off_rev(T, t)
    M[2, 2] = -bp_2 * _phi_mirror(T, t) + _psi(t) * (bj_2 * n1_2**2 + bm_2)

    d = np.zeros(3)
    d[0] = a_tau_0 * bp_0 * dy * n1_0 - a_0 * bp_0 * _couple_avg(T, t) + b_0 * dy * n2_0
    d[1] = -a_tau_1 * bp_1 * dx * n2_1 + a_1 * bp_1 * -_phi(L) + b_1 * dx * n1_1
    d[2] = a_tau_2 * bp_2 * dy * n1_2 + a_2 * bp_2 * _couple_avg(T, t) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 4] = (
        T
        * (
            bj_0 * dx * n1_0**2
            - bj_0 * dy * n1_0 * n2_0 * T
            - bj_0 * dy * n1_0 * n2_0
            + bm_0 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[0, 5] = -(
        bj_0 * dx * n1_0**2 * L * T
        + bj_0 * dx * n1_0**2 * L
        - bj_0 * dy * n1_0 * n2_0 * L * T**2
        - bj_0 * dy * n1_0 * n2_0 * L * T
        + bj_0 * dy * n1_0 * n2_0 * T
        + bm_0 * dx * L * T
        + bm_0 * dx * L
    ) / (dx * L * T)  # (+0, +0)
    N[0, 6] = bp_0 * _couple_avg(T, t)  # (+0, +1)
    N[0, 9] = bj_0 * dy * n1_0 * n2_0 * T / dx  # (+1, -1)
    N[0, 10] = -bj_0 * dy * n1_0 * n2_0 * (L * T + L + T) / (dx * (L + 1))  # (+1, +0)
    N[1, 0] = -bp_1 * (L - 1) / (L - 2)  # (-2, +0)
    N[1, 1] = bp_1 * (L - 2) / (L - 1)  # (-1, +0)
    N[1, 4] = bj_1 * dx * n1_1 * n2_1 * (L * T + L + T) / (dy * (T + 1))  # (+0, -1)
    N[1, 5] = -(
        bj_1 * dx * n1_1 * n2_1 * L**2 * T
        + bj_1 * dx * n1_1 * n2_1 * L * T
        - bj_1 * dx * n1_1 * n2_1 * L
        - bj_1 * dy * n2_1**2 * L * T
        - bj_1 * dy * n2_1**2 * T
        - bm_1 * dy * L * T
        - bm_1 * dy * T
    ) / (dy * L * T)  # (+0, +0)
    N[1, 9] = -bj_1 * dx * n1_1 * n2_1 * L / dy  # (+1, -1)
    N[1, 10] = (
        L
        * (
            bj_1 * dx * n1_1 * n2_1 * L
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            - bm_1 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)
    N[2, 2] = (1 / 2) * bj_2 * dy * n1_2 * n2_2 * (2 * t + 1) / dx  # (-1, +2)
    N[2, 3] = -bj_2 * dy * n1_2 * n2_2 * t / dx  # (-1, +3)
    N[2, 6] = -bp_2 * _couple_avg(T, t)  # (+0, +1)
    N[2, 7] = (
        bj_2 * dx * n1_2**2 * t
        + bj_2 * dx * n1_2**2
        - bj_2 * dy * n1_2 * n2_2 * t**2
        + bm_2 * dx * t
        + bm_2 * dx
    ) / (dx * t)  # (+0, +2)
    N[2, 8] = (
        -t
        * (
            bj_2 * dx * n1_2**2
            - bj_2 * dy * n1_2 * n2_2 * t
            - bj_2 * dy * n1_2 * n2_2
            + bm_2 * dx
        )
        / (dx * (t + 1))
    )  # (+0, +3)
    N[2, 11] = -1 / 2 * bj_2 * dy * n1_2 * n2_2 / dx  # (+1, +2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc3_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 3, eta > 0.  Row 2 is extra (T,L corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "T")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "L")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, +0),
        (-1, +2),
        (-1, +3),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+0, +3),
        (+1, -1),
        (+1, +0),
        (+1, +2),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bm_0 * _phi_coupled(T, t) - _psi(T) * (bj_0 * n1_0**2 - bp_0)
    M[0, 1] = -bj_0 * dy * n1_0 * n2_0 / (dx * L * (L + 1))
    M[0, 2] = -bm_0 * _couple_off_fwd(T, t)
    M[1, 0] = bj_1 * dx * n1_1 * n2_1 / (dy * T * (T + 1))
    M[1, 1] = bm_1 * -_phi(L) + _psi(L) * (bj_1 * n2_1**2 - bp_1)
    M[2, 0] = bm_2 * _couple_off_rev(T, t)
    M[2, 2] = bm_2 * _phi_mirror(T, t) + _psi(t) * (bj_2 * n1_2**2 - bp_2)

    d = np.zeros(3)
    d[0] = a_tau_0 * bm_0 * dy * n1_0 - a_0 * bm_0 * _couple_avg(T, t) + b_0 * dy * n2_0
    d[1] = -a_tau_1 * bm_1 * dx * n2_1 + a_1 * bm_1 * -_phi(L) + b_1 * dx * n1_1
    d[2] = a_tau_2 * bm_2 * dy * n1_2 + a_2 * bm_2 * _couple_avg(T, t) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 4] = (
        T
        * (
            bj_0 * dx * n1_0**2
            - bj_0 * dy * n1_0 * n2_0 * T
            - bj_0 * dy * n1_0 * n2_0
            - bp_0 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[0, 5] = -(
        bj_0 * dx * n1_0**2 * L * T
        + bj_0 * dx * n1_0**2 * L
        - bj_0 * dy * n1_0 * n2_0 * L * T**2
        - bj_0 * dy * n1_0 * n2_0 * L * T
        + bj_0 * dy * n1_0 * n2_0 * T
        - bp_0 * dx * L * T
        - bp_0 * dx * L
    ) / (dx * L * T)  # (+0, +0)
    N[0, 6] = -bm_0 * _couple_avg(T, t)  # (+0, +1)
    N[0, 9] = bj_0 * dy * n1_0 * n2_0 * T / dx  # (+1, -1)
    N[0, 10] = -bj_0 * dy * n1_0 * n2_0 * (L * T + L + T) / (dx * (L + 1))  # (+1, +0)
    N[1, 0] = bm_1 * (L - 1) / (L - 2)  # (-2, +0)
    N[1, 1] = -bm_1 * (L - 2) / (L - 1)  # (-1, +0)
    N[1, 4] = bj_1 * dx * n1_1 * n2_1 * (L * T + L + T) / (dy * (T + 1))  # (+0, -1)
    N[1, 5] = -(
        bj_1 * dx * n1_1 * n2_1 * L**2 * T
        + bj_1 * dx * n1_1 * n2_1 * L * T
        - bj_1 * dx * n1_1 * n2_1 * L
        - bj_1 * dy * n2_1**2 * L * T
        - bj_1 * dy * n2_1**2 * T
        + bp_1 * dy * L * T
        + bp_1 * dy * T
    ) / (dy * L * T)  # (+0, +0)
    N[1, 9] = -bj_1 * dx * n1_1 * n2_1 * L / dy  # (+1, -1)
    N[1, 10] = (
        L
        * (
            bj_1 * dx * n1_1 * n2_1 * L
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            + bp_1 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)
    N[2, 2] = (1 / 2) * bj_2 * dy * n1_2 * n2_2 * (2 * t + 1) / dx  # (-1, +2)
    N[2, 3] = -bj_2 * dy * n1_2 * n2_2 * t / dx  # (-1, +3)
    N[2, 6] = bm_2 * _couple_avg(T, t)  # (+0, +1)
    N[2, 7] = (
        bj_2 * dx * n1_2**2 * t
        + bj_2 * dx * n1_2**2
        - bj_2 * dy * n1_2 * n2_2 * t**2
        - bp_2 * dx * t
        - bp_2 * dx
    ) / (dx * t)  # (+0, +2)
    N[2, 8] = (
        -t
        * (
            bj_2 * dx * n1_2**2
            - bj_2 * dy * n1_2 * n2_2 * t
            - bj_2 * dy * n1_2 * n2_2
            - bp_2 * dx
        )
        / (dx * (t + 1))
    )  # (+0, +3)
    N[2, 11] = -1 / 2 * bj_2 * dy * n1_2 * n2_2 / dx  # (+1, +2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc4_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 4, eta < 0.  Row 2 is extra (T,L corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "T")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "L")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-3, -1),
        (-3, +0),
        (-2, -1),
        (-2, +0),
        (-2, +1),
        (-1, +0),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, -1),
        (+1, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bp_0 * -_phi(T) - _psi(T) * (bj_0 * n1_0**2 + bm_0)
    M[0, 1] = -bj_0 * dy * n1_0 * n2_0 / (dx * L * (L + 1))
    M[1, 0] = bj_1 * dx * n1_1 * n2_1 / (dy * T * (T + 1))
    M[1, 1] = -bp_1 * _phi_coupled(L, l) + _psi(L) * (bj_1 * n2_1**2 + bm_1)
    M[1, 2] = -bp_1 * _couple_off_fwd(L, l)
    M[2, 2] = bp_2 * _couple_off_rev(L, l)

    d = np.zeros(3)
    d[0] = a_tau_0 * bp_0 * dy * n1_0 - a_0 * bp_0 * -_phi(T) + b_0 * dy * n2_0
    d[1] = (
        -a_tau_1 * bp_1 * dx * n2_1 + a_1 * bp_1 * _couple_avg(L, l) + b_1 * dx * n1_1
    )
    d[2] = (
        -a_tau_2 * bp_2 * dx * n2_2 - a_2 * bp_2 * _couple_avg(L, l) + b_2 * dx * n1_2
    )

    N = np.zeros((3, len(offsets)))
    N[0, 6] = (
        T
        * (
            bj_0 * dx * n1_0**2
            - bj_0 * dy * n1_0 * n2_0 * T
            - bj_0 * dy * n1_0 * n2_0
            + bm_0 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[0, 7] = -(
        bj_0 * dx * n1_0**2 * L * T
        + bj_0 * dx * n1_0**2 * L
        - bj_0 * dy * n1_0 * n2_0 * L * T**2
        - bj_0 * dy * n1_0 * n2_0 * L * T
        + bj_0 * dy * n1_0 * n2_0 * T
        + bm_0 * dx * L * T
        + bm_0 * dx * L
    ) / (dx * L * T)  # (+0, +0)
    N[0, 8] = -bp_0 * (T - 2) / (T - 1)  # (+0, +1)
    N[0, 9] = bp_0 * (T - 1) / (T - 2)  # (+0, +2)
    N[0, 10] = bj_0 * dy * n1_0 * n2_0 * T / dx  # (+1, -1)
    N[0, 11] = -bj_0 * dy * n1_0 * n2_0 * (L * T + L + T) / (dx * (L + 1))  # (+1, +0)
    N[1, 5] = -bp_1 * _couple_avg(L, l)  # (-1, +0)
    N[1, 6] = bj_1 * dx * n1_1 * n2_1 * (L * T + L + T) / (dy * (T + 1))  # (+0, -1)
    N[1, 7] = -(
        bj_1 * dx * n1_1 * n2_1 * L**2 * T
        + bj_1 * dx * n1_1 * n2_1 * L * T
        - bj_1 * dx * n1_1 * n2_1 * L
        - bj_1 * dy * n2_1**2 * L * T
        - bj_1 * dy * n2_1**2 * T
        - bm_1 * dy * L * T
        - bm_1 * dy * T
    ) / (dy * L * T)  # (+0, +0)
    N[1, 10] = -bj_1 * dx * n1_1 * n2_1 * L / dy  # (+1, -1)
    N[1, 11] = (
        L
        * (
            bj_1 * dx * n1_1 * n2_1 * L
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            - bm_1 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)
    N[2, 0] = -bj_2 * dx * n1_2 * n2_2 * l / dy  # (-3, -1)
    N[2, 1] = (
        l
        * (
            bj_2 * dx * n1_2 * n2_2 * l
            + bj_2 * dx * n1_2 * n2_2
            + bj_2 * dy * n2_2**2
            + bm_2 * dy
        )
        / (dy * (l + 1))
    )  # (-3, +0)
    N[2, 2] = (1 / 2) * bj_2 * dx * n1_2 * n2_2 * (2 * l + 1) / dy  # (-2, -1)
    N[2, 3] = -(
        bj_2 * dx * n1_2 * n2_2 * l**2
        + bj_2 * dy * n2_2**2 * l
        + bj_2 * dy * n2_2**2
        + bm_2 * dy * l
        + bm_2 * dy
    ) / (dy * l)  # (-2, +0)
    N[2, 4] = -1 / 2 * bj_2 * dx * n1_2 * n2_2 / dy  # (-2, +1)
    N[2, 5] = bp_2 * _couple_avg(L, l)  # (-1, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc4_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 4, eta > 0.  Row 2 is extra (T,L corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "T")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "L")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-3, -1),
        (-3, +0),
        (-2, -1),
        (-2, +0),
        (-2, +1),
        (-1, +0),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, -1),
        (+1, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bm_0 * -_phi(T) - _psi(T) * (bj_0 * n1_0**2 - bp_0)
    M[0, 1] = -bj_0 * dy * n1_0 * n2_0 / (dx * L * (L + 1))
    M[1, 0] = bj_1 * dx * n1_1 * n2_1 / (dy * T * (T + 1))
    M[1, 1] = bm_1 * _phi_coupled(L, l) + _psi(L) * (bj_1 * n2_1**2 - bp_1)
    M[1, 2] = bm_1 * _couple_off_fwd(L, l)
    M[2, 2] = -bm_2 * _couple_off_rev(L, l)

    d = np.zeros(3)
    d[0] = a_tau_0 * bm_0 * dy * n1_0 - a_0 * bm_0 * -_phi(T) + b_0 * dy * n2_0
    d[1] = (
        -a_tau_1 * bm_1 * dx * n2_1 + a_1 * bm_1 * _couple_avg(L, l) + b_1 * dx * n1_1
    )
    d[2] = (
        -a_tau_2 * bm_2 * dx * n2_2 - a_2 * bm_2 * _couple_avg(L, l) + b_2 * dx * n1_2
    )

    N = np.zeros((3, len(offsets)))
    N[0, 6] = (
        T
        * (
            bj_0 * dx * n1_0**2
            - bj_0 * dy * n1_0 * n2_0 * T
            - bj_0 * dy * n1_0 * n2_0
            - bp_0 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[0, 7] = -(
        bj_0 * dx * n1_0**2 * L * T
        + bj_0 * dx * n1_0**2 * L
        - bj_0 * dy * n1_0 * n2_0 * L * T**2
        - bj_0 * dy * n1_0 * n2_0 * L * T
        + bj_0 * dy * n1_0 * n2_0 * T
        - bp_0 * dx * L * T
        - bp_0 * dx * L
    ) / (dx * L * T)  # (+0, +0)
    N[0, 8] = bm_0 * (T - 2) / (T - 1)  # (+0, +1)
    N[0, 9] = -bm_0 * (T - 1) / (T - 2)  # (+0, +2)
    N[0, 10] = bj_0 * dy * n1_0 * n2_0 * T / dx  # (+1, -1)
    N[0, 11] = -bj_0 * dy * n1_0 * n2_0 * (L * T + L + T) / (dx * (L + 1))  # (+1, +0)
    N[1, 5] = bm_1 * _couple_avg(L, l)  # (-1, +0)
    N[1, 6] = bj_1 * dx * n1_1 * n2_1 * (L * T + L + T) / (dy * (T + 1))  # (+0, -1)
    N[1, 7] = -(
        bj_1 * dx * n1_1 * n2_1 * L**2 * T
        + bj_1 * dx * n1_1 * n2_1 * L * T
        - bj_1 * dx * n1_1 * n2_1 * L
        - bj_1 * dy * n2_1**2 * L * T
        - bj_1 * dy * n2_1**2 * T
        + bp_1 * dy * L * T
        + bp_1 * dy * T
    ) / (dy * L * T)  # (+0, +0)
    N[1, 10] = -bj_1 * dx * n1_1 * n2_1 * L / dy  # (+1, -1)
    N[1, 11] = (
        L
        * (
            bj_1 * dx * n1_1 * n2_1 * L
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            + bp_1 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)
    N[2, 0] = -bj_2 * dx * n1_2 * n2_2 * l / dy  # (-3, -1)
    N[2, 1] = (
        l
        * (
            bj_2 * dx * n1_2 * n2_2 * l
            + bj_2 * dx * n1_2 * n2_2
            + bj_2 * dy * n2_2**2
            - bp_2 * dy
        )
        / (dy * (l + 1))
    )  # (-3, +0)
    N[2, 2] = (1 / 2) * bj_2 * dx * n1_2 * n2_2 * (2 * l + 1) / dy  # (-2, -1)
    N[2, 3] = -(
        bj_2 * dx * n1_2 * n2_2 * l**2
        + bj_2 * dy * n2_2**2 * l
        + bj_2 * dy * n2_2**2
        - bp_2 * dy * l
        - bp_2 * dy
    ) / (dy * l)  # (-2, +0)
    N[2, 4] = -1 / 2 * bj_2 * dx * n1_2 * n2_2 / dy  # (-2, +1)
    N[2, 5] = -bm_2 * _couple_avg(L, l)  # (-1, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc5_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 5, eta < 0.  Row 2 is extra (L,B corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "L")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "B")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-3, -1),
        (-3, +0),
        (-2, -1),
        (-2, +0),
        (-2, +1),
        (-1, +0),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, +0),
        (+1, +1),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bp_0 * _phi_coupled(L, l) + _psi(L) * (bj_0 * n2_0**2 + bm_0)
    M[0, 1] = -bj_0 * dx * n1_0 * n2_0 / (dy * B * (B + 1))
    M[0, 2] = -bp_0 * _couple_off_fwd(L, l)
    M[1, 0] = -bj_1 * dy * n1_1 * n2_1 / (dx * L * (L + 1))
    M[1, 1] = -bp_1 * -_phi(B) + _psi(B) * (bj_1 * n1_1**2 + bm_1)
    M[2, 0] = bp_2 * _couple_off_rev(L, l)
    M[2, 2] = bp_2 * _phi_mirror(L, l) - _psi(l) * (bj_2 * n2_2**2 + bm_2)

    d = np.zeros(3)
    d[0] = (
        -a_tau_0 * bp_0 * dx * n2_0 + a_0 * bp_0 * _couple_avg(L, l) + b_0 * dx * n1_0
    )
    d[1] = a_tau_1 * bp_1 * dy * n1_1 + a_1 * bp_1 * -_phi(B) + b_1 * dy * n2_1
    d[2] = (
        -a_tau_2 * bp_2 * dx * n2_2 - a_2 * bp_2 * _couple_avg(L, l) + b_2 * dx * n1_2
    )

    N = np.zeros((3, len(offsets)))
    N[0, 5] = -bp_0 * _couple_avg(L, l)  # (-1, +0)
    N[0, 8] = (
        bj_0 * dx * n1_0 * n2_0 * B * L**2
        + bj_0 * dx * n1_0 * n2_0 * B * L
        - bj_0 * dx * n1_0 * n2_0 * L
        + bj_0 * dy * n2_0**2 * B * L
        + bj_0 * dy * n2_0**2 * B
        + bm_0 * dy * B * L
        + bm_0 * dy * B
    ) / (dy * B * L)  # (+0, +0)
    N[0, 9] = -bj_0 * dx * n1_0 * n2_0 * (B * L + B + L) / (dy * (B + 1))  # (+0, +1)
    N[0, 10] = (
        -L
        * (
            bj_0 * dx * n1_0 * n2_0 * L
            + bj_0 * dx * n1_0 * n2_0
            + bj_0 * dy * n2_0**2
            + bm_0 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)
    N[0, 11] = bj_0 * dx * n1_0 * n2_0 * L / dy  # (+1, +1)
    N[1, 6] = -bp_1 * (B - 1) / (B - 2)  # (+0, -2)
    N[1, 7] = bp_1 * (B - 2) / (B - 1)  # (+0, -1)
    N[1, 8] = (
        bj_1 * dx * n1_1**2 * B * L
        + bj_1 * dx * n1_1**2 * L
        + bj_1 * dy * n1_1 * n2_1 * B**2 * L
        + bj_1 * dy * n1_1 * n2_1 * B * L
        - bj_1 * dy * n1_1 * n2_1 * B
        + bm_1 * dx * B * L
        + bm_1 * dx * L
    ) / (dx * B * L)  # (+0, +0)
    N[1, 9] = (
        -B
        * (
            bj_1 * dx * n1_1**2
            + bj_1 * dy * n1_1 * n2_1 * B
            + bj_1 * dy * n1_1 * n2_1
            + bm_1 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[1, 10] = -bj_1 * dy * n1_1 * n2_1 * (B * L + B + L) / (dx * (L + 1))  # (+1, +0)
    N[1, 11] = bj_1 * dy * n1_1 * n2_1 * B / dx  # (+1, +1)
    N[2, 0] = -bj_2 * dx * n1_2 * n2_2 * l / dy  # (-3, -1)
    N[2, 1] = (
        l
        * (
            bj_2 * dx * n1_2 * n2_2 * l
            + bj_2 * dx * n1_2 * n2_2
            + bj_2 * dy * n2_2**2
            + bm_2 * dy
        )
        / (dy * (l + 1))
    )  # (-3, +0)
    N[2, 2] = (1 / 2) * bj_2 * dx * n1_2 * n2_2 * (2 * l + 1) / dy  # (-2, -1)
    N[2, 3] = -(
        bj_2 * dx * n1_2 * n2_2 * l**2
        + bj_2 * dy * n2_2**2 * l
        + bj_2 * dy * n2_2**2
        + bm_2 * dy * l
        + bm_2 * dy
    ) / (dy * l)  # (-2, +0)
    N[2, 4] = -1 / 2 * bj_2 * dx * n1_2 * n2_2 / dy  # (-2, +1)
    N[2, 5] = bp_2 * _couple_avg(L, l)  # (-1, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc5_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 5, eta > 0.  Row 2 is extra (L,B corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "L")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "B")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-3, -1),
        (-3, +0),
        (-2, -1),
        (-2, +0),
        (-2, +1),
        (-1, +0),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, +0),
        (+1, +1),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bm_0 * _phi_coupled(L, l) + _psi(L) * (bj_0 * n2_0**2 - bp_0)
    M[0, 1] = -bj_0 * dx * n1_0 * n2_0 / (dy * B * (B + 1))
    M[0, 2] = bm_0 * _couple_off_fwd(L, l)
    M[1, 0] = -bj_1 * dy * n1_1 * n2_1 / (dx * L * (L + 1))
    M[1, 1] = bm_1 * -_phi(B) + _psi(B) * (bj_1 * n1_1**2 - bp_1)
    M[2, 0] = -bm_2 * _couple_off_rev(L, l)
    M[2, 2] = -bm_2 * _phi_mirror(L, l) - _psi(l) * (bj_2 * n2_2**2 - bp_2)

    d = np.zeros(3)
    d[0] = (
        -a_tau_0 * bm_0 * dx * n2_0 + a_0 * bm_0 * _couple_avg(L, l) + b_0 * dx * n1_0
    )
    d[1] = a_tau_1 * bm_1 * dy * n1_1 + a_1 * bm_1 * -_phi(B) + b_1 * dy * n2_1
    d[2] = (
        -a_tau_2 * bm_2 * dx * n2_2 - a_2 * bm_2 * _couple_avg(L, l) + b_2 * dx * n1_2
    )

    N = np.zeros((3, len(offsets)))
    N[0, 5] = bm_0 * _couple_avg(L, l)  # (-1, +0)
    N[0, 8] = (
        bj_0 * dx * n1_0 * n2_0 * B * L**2
        + bj_0 * dx * n1_0 * n2_0 * B * L
        - bj_0 * dx * n1_0 * n2_0 * L
        + bj_0 * dy * n2_0**2 * B * L
        + bj_0 * dy * n2_0**2 * B
        - bp_0 * dy * B * L
        - bp_0 * dy * B
    ) / (dy * B * L)  # (+0, +0)
    N[0, 9] = -bj_0 * dx * n1_0 * n2_0 * (B * L + B + L) / (dy * (B + 1))  # (+0, +1)
    N[0, 10] = (
        -L
        * (
            bj_0 * dx * n1_0 * n2_0 * L
            + bj_0 * dx * n1_0 * n2_0
            + bj_0 * dy * n2_0**2
            - bp_0 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)
    N[0, 11] = bj_0 * dx * n1_0 * n2_0 * L / dy  # (+1, +1)
    N[1, 6] = bm_1 * (B - 1) / (B - 2)  # (+0, -2)
    N[1, 7] = -bm_1 * (B - 2) / (B - 1)  # (+0, -1)
    N[1, 8] = (
        bj_1 * dx * n1_1**2 * B * L
        + bj_1 * dx * n1_1**2 * L
        + bj_1 * dy * n1_1 * n2_1 * B**2 * L
        + bj_1 * dy * n1_1 * n2_1 * B * L
        - bj_1 * dy * n1_1 * n2_1 * B
        - bp_1 * dx * B * L
        - bp_1 * dx * L
    ) / (dx * B * L)  # (+0, +0)
    N[1, 9] = (
        -B
        * (
            bj_1 * dx * n1_1**2
            + bj_1 * dy * n1_1 * n2_1 * B
            + bj_1 * dy * n1_1 * n2_1
            - bp_1 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[1, 10] = -bj_1 * dy * n1_1 * n2_1 * (B * L + B + L) / (dx * (L + 1))  # (+1, +0)
    N[1, 11] = bj_1 * dy * n1_1 * n2_1 * B / dx  # (+1, +1)
    N[2, 0] = -bj_2 * dx * n1_2 * n2_2 * l / dy  # (-3, -1)
    N[2, 1] = (
        l
        * (
            bj_2 * dx * n1_2 * n2_2 * l
            + bj_2 * dx * n1_2 * n2_2
            + bj_2 * dy * n2_2**2
            - bp_2 * dy
        )
        / (dy * (l + 1))
    )  # (-3, +0)
    N[2, 2] = (1 / 2) * bj_2 * dx * n1_2 * n2_2 * (2 * l + 1) / dy  # (-2, -1)
    N[2, 3] = -(
        bj_2 * dx * n1_2 * n2_2 * l**2
        + bj_2 * dy * n2_2**2 * l
        + bj_2 * dy * n2_2**2
        - bp_2 * dy * l
        - bp_2 * dy
    ) / (dy * l)  # (-2, +0)
    N[2, 4] = -1 / 2 * bj_2 * dx * n1_2 * n2_2 / dy  # (-2, +1)
    N[2, 5] = -bm_2 * _couple_avg(L, l)  # (-1, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc6_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 6, eta < 0.  Row 2 is extra (L,B corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "L")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "B")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, -3),
        (-1, -2),
        (-1, +0),
        (+0, -3),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, -2),
        (+1, +0),
        (+1, +1),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bp_0 * -_phi(L) + _psi(L) * (bj_0 * n2_0**2 + bm_0)
    M[0, 1] = -bj_0 * dx * n1_0 * n2_0 / (dy * B * (B + 1))
    M[1, 0] = -bj_1 * dy * n1_1 * n2_1 / (dx * L * (L + 1))
    M[1, 1] = -bp_1 * _phi_coupled(B, b) + _psi(B) * (bj_1 * n1_1**2 + bm_1)
    M[1, 2] = -bp_1 * _couple_off_fwd(B, b)
    M[2, 2] = bp_2 * _couple_off_rev(B, b)

    d = np.zeros(3)
    d[0] = -a_tau_0 * bp_0 * dx * n2_0 + a_0 * bp_0 * -_phi(L) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bp_1 * dy * n1_1 + a_1 * bp_1 * _couple_avg(B, b) + b_1 * dy * n2_1
    d[2] = a_tau_2 * bp_2 * dy * n1_2 - a_2 * bp_2 * _couple_avg(B, b) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 0] = -bp_0 * (L - 1) / (L - 2)  # (-2, +0)
    N[0, 3] = bp_0 * (L - 2) / (L - 1)  # (-1, +0)
    N[0, 7] = (
        bj_0 * dx * n1_0 * n2_0 * B * L**2
        + bj_0 * dx * n1_0 * n2_0 * B * L
        - bj_0 * dx * n1_0 * n2_0 * L
        + bj_0 * dy * n2_0**2 * B * L
        + bj_0 * dy * n2_0**2 * B
        + bm_0 * dy * B * L
        + bm_0 * dy * B
    ) / (dy * B * L)  # (+0, +0)
    N[0, 8] = -bj_0 * dx * n1_0 * n2_0 * (B * L + B + L) / (dy * (B + 1))  # (+0, +1)
    N[0, 10] = (
        -L
        * (
            bj_0 * dx * n1_0 * n2_0 * L
            + bj_0 * dx * n1_0 * n2_0
            + bj_0 * dy * n2_0**2
            + bm_0 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)
    N[0, 11] = bj_0 * dx * n1_0 * n2_0 * L / dy  # (+1, +1)
    N[1, 6] = -bp_1 * _couple_avg(B, b)  # (+0, -1)
    N[1, 7] = (
        bj_1 * dx * n1_1**2 * B * L
        + bj_1 * dx * n1_1**2 * L
        + bj_1 * dy * n1_1 * n2_1 * B**2 * L
        + bj_1 * dy * n1_1 * n2_1 * B * L
        - bj_1 * dy * n1_1 * n2_1 * B
        + bm_1 * dx * B * L
        + bm_1 * dx * L
    ) / (dx * B * L)  # (+0, +0)
    N[1, 8] = (
        -B
        * (
            bj_1 * dx * n1_1**2
            + bj_1 * dy * n1_1 * n2_1 * B
            + bj_1 * dy * n1_1 * n2_1
            + bm_1 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[1, 10] = -bj_1 * dy * n1_1 * n2_1 * (B * L + B + L) / (dx * (L + 1))  # (+1, +0)
    N[1, 11] = bj_1 * dy * n1_1 * n2_1 * B / dx  # (+1, +1)
    N[2, 1] = -bj_2 * dy * n1_2 * n2_2 * b / dx  # (-1, -3)
    N[2, 2] = (1 / 2) * bj_2 * dy * n1_2 * n2_2 * (2 * b + 1) / dx  # (-1, -2)
    N[2, 4] = (
        b
        * (
            bj_2 * dx * n1_2**2
            + bj_2 * dy * n1_2 * n2_2 * b
            + bj_2 * dy * n1_2 * n2_2
            + bm_2 * dx
        )
        / (dx * (b + 1))
    )  # (+0, -3)
    N[2, 5] = -(
        bj_2 * dx * n1_2**2 * b
        + bj_2 * dx * n1_2**2
        + bj_2 * dy * n1_2 * n2_2 * b**2
        + bm_2 * dx * b
        + bm_2 * dx
    ) / (dx * b)  # (+0, -2)
    N[2, 6] = bp_2 * _couple_avg(B, b)  # (+0, -1)
    N[2, 9] = -1 / 2 * bj_2 * dy * n1_2 * n2_2 / dx  # (+1, -2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc6_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 6, eta > 0.  Row 2 is extra (L,B corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "L")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "B")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, -3),
        (-1, -2),
        (-1, +0),
        (+0, -3),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, -2),
        (+1, +0),
        (+1, +1),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bm_0 * -_phi(L) + _psi(L) * (bj_0 * n2_0**2 - bp_0)
    M[0, 1] = -bj_0 * dx * n1_0 * n2_0 / (dy * B * (B + 1))
    M[1, 0] = -bj_1 * dy * n1_1 * n2_1 / (dx * L * (L + 1))
    M[1, 1] = bm_1 * _phi_coupled(B, b) + _psi(B) * (bj_1 * n1_1**2 - bp_1)
    M[1, 2] = bm_1 * _couple_off_fwd(B, b)
    M[2, 2] = -bm_2 * _couple_off_rev(B, b)

    d = np.zeros(3)
    d[0] = -a_tau_0 * bm_0 * dx * n2_0 + a_0 * bm_0 * -_phi(L) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bm_1 * dy * n1_1 + a_1 * bm_1 * _couple_avg(B, b) + b_1 * dy * n2_1
    d[2] = a_tau_2 * bm_2 * dy * n1_2 - a_2 * bm_2 * _couple_avg(B, b) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 0] = bm_0 * (L - 1) / (L - 2)  # (-2, +0)
    N[0, 3] = -bm_0 * (L - 2) / (L - 1)  # (-1, +0)
    N[0, 7] = (
        bj_0 * dx * n1_0 * n2_0 * B * L**2
        + bj_0 * dx * n1_0 * n2_0 * B * L
        - bj_0 * dx * n1_0 * n2_0 * L
        + bj_0 * dy * n2_0**2 * B * L
        + bj_0 * dy * n2_0**2 * B
        - bp_0 * dy * B * L
        - bp_0 * dy * B
    ) / (dy * B * L)  # (+0, +0)
    N[0, 8] = -bj_0 * dx * n1_0 * n2_0 * (B * L + B + L) / (dy * (B + 1))  # (+0, +1)
    N[0, 10] = (
        -L
        * (
            bj_0 * dx * n1_0 * n2_0 * L
            + bj_0 * dx * n1_0 * n2_0
            + bj_0 * dy * n2_0**2
            - bp_0 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)
    N[0, 11] = bj_0 * dx * n1_0 * n2_0 * L / dy  # (+1, +1)
    N[1, 6] = bm_1 * _couple_avg(B, b)  # (+0, -1)
    N[1, 7] = (
        bj_1 * dx * n1_1**2 * B * L
        + bj_1 * dx * n1_1**2 * L
        + bj_1 * dy * n1_1 * n2_1 * B**2 * L
        + bj_1 * dy * n1_1 * n2_1 * B * L
        - bj_1 * dy * n1_1 * n2_1 * B
        - bp_1 * dx * B * L
        - bp_1 * dx * L
    ) / (dx * B * L)  # (+0, +0)
    N[1, 8] = (
        -B
        * (
            bj_1 * dx * n1_1**2
            + bj_1 * dy * n1_1 * n2_1 * B
            + bj_1 * dy * n1_1 * n2_1
            - bp_1 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[1, 10] = -bj_1 * dy * n1_1 * n2_1 * (B * L + B + L) / (dx * (L + 1))  # (+1, +0)
    N[1, 11] = bj_1 * dy * n1_1 * n2_1 * B / dx  # (+1, +1)
    N[2, 1] = -bj_2 * dy * n1_2 * n2_2 * b / dx  # (-1, -3)
    N[2, 2] = (1 / 2) * bj_2 * dy * n1_2 * n2_2 * (2 * b + 1) / dx  # (-1, -2)
    N[2, 4] = (
        b
        * (
            bj_2 * dx * n1_2**2
            + bj_2 * dy * n1_2 * n2_2 * b
            + bj_2 * dy * n1_2 * n2_2
            - bp_2 * dx
        )
        / (dx * (b + 1))
    )  # (+0, -3)
    N[2, 5] = -(
        bj_2 * dx * n1_2**2 * b
        + bj_2 * dx * n1_2**2
        + bj_2 * dy * n1_2 * n2_2 * b**2
        - bp_2 * dx * b
        - bp_2 * dx
    ) / (dx * b)  # (+0, -2)
    N[2, 6] = -bm_2 * _couple_avg(B, b)  # (+0, -1)
    N[2, 9] = -1 / 2 * bj_2 * dy * n1_2 * n2_2 / dx  # (+1, -2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc7_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 7, eta < 0.  Row 2 is extra (B,R corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "B")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "R")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, -3),
        (-1, -2),
        (-1, +0),
        (-1, +1),
        (+0, -3),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, -2),
        (+1, +0),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bp_0 * _phi_coupled(B, b) + _psi(B) * (bj_0 * n1_0**2 + bm_0)
    M[0, 1] = bj_0 * dy * n1_0 * n2_0 / (dx * R * (R + 1))
    M[0, 2] = -bp_0 * _couple_off_fwd(B, b)
    M[1, 0] = -bj_1 * dx * n1_1 * n2_1 / (dy * B * (B + 1))
    M[1, 1] = bp_1 * -_phi(R) - _psi(R) * (bj_1 * n2_1**2 + bm_1)
    M[2, 0] = bp_2 * _couple_off_rev(B, b)
    M[2, 2] = bp_2 * _phi_mirror(B, b) - _psi(b) * (bj_2 * n1_2**2 + bm_2)

    d = np.zeros(3)
    d[0] = a_tau_0 * bp_0 * dy * n1_0 + a_0 * bp_0 * _couple_avg(B, b) + b_0 * dy * n2_0
    d[1] = -a_tau_1 * bp_1 * dx * n2_1 - a_1 * bp_1 * -_phi(R) + b_1 * dx * n1_1
    d[2] = a_tau_2 * bp_2 * dy * n1_2 - a_2 * bp_2 * _couple_avg(B, b) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 2] = bj_0 * dy * n1_0 * n2_0 * (B * R + B + R) / (dx * (R + 1))  # (-1, +0)
    N[0, 3] = -bj_0 * dy * n1_0 * n2_0 * B / dx  # (-1, +1)
    N[0, 6] = -bp_0 * _couple_avg(B, b)  # (+0, -1)
    N[0, 7] = (
        bj_0 * dx * n1_0**2 * B * R
        + bj_0 * dx * n1_0**2 * R
        - bj_0 * dy * n1_0 * n2_0 * B**2 * R
        - bj_0 * dy * n1_0 * n2_0 * B * R
        + bj_0 * dy * n1_0 * n2_0 * B
        + bm_0 * dx * B * R
        + bm_0 * dx * R
    ) / (dx * B * R)  # (+0, +0)
    N[0, 8] = (
        -B
        * (
            bj_0 * dx * n1_0**2
            - bj_0 * dy * n1_0 * n2_0 * B
            - bj_0 * dy * n1_0 * n2_0
            + bm_0 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[1, 2] = (
        -R
        * (
            bj_1 * dx * n1_1 * n2_1 * R
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            - bm_1 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[1, 3] = bj_1 * dx * n1_1 * n2_1 * R / dy  # (-1, +1)
    N[1, 7] = (
        bj_1 * dx * n1_1 * n2_1 * B * R**2
        + bj_1 * dx * n1_1 * n2_1 * B * R
        - bj_1 * dx * n1_1 * n2_1 * R
        - bj_1 * dy * n2_1**2 * B * R
        - bj_1 * dy * n2_1**2 * B
        - bm_1 * dy * B * R
        - bm_1 * dy * B
    ) / (dy * B * R)  # (+0, +0)
    N[1, 8] = -bj_1 * dx * n1_1 * n2_1 * (B * R + B + R) / (dy * (B + 1))  # (+0, +1)
    N[1, 10] = -bp_1 * (R - 2) / (R - 1)  # (+1, +0)
    N[1, 11] = bp_1 * (R - 1) / (R - 2)  # (+2, +0)
    N[2, 0] = -bj_2 * dy * n1_2 * n2_2 * b / dx  # (-1, -3)
    N[2, 1] = (1 / 2) * bj_2 * dy * n1_2 * n2_2 * (2 * b + 1) / dx  # (-1, -2)
    N[2, 4] = (
        b
        * (
            bj_2 * dx * n1_2**2
            + bj_2 * dy * n1_2 * n2_2 * b
            + bj_2 * dy * n1_2 * n2_2
            + bm_2 * dx
        )
        / (dx * (b + 1))
    )  # (+0, -3)
    N[2, 5] = -(
        bj_2 * dx * n1_2**2 * b
        + bj_2 * dx * n1_2**2
        + bj_2 * dy * n1_2 * n2_2 * b**2
        + bm_2 * dx * b
        + bm_2 * dx
    ) / (dx * b)  # (+0, -2)
    N[2, 6] = bp_2 * _couple_avg(B, b)  # (+0, -1)
    N[2, 9] = -1 / 2 * bj_2 * dy * n1_2 * n2_2 / dx  # (+1, -2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc7_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 7, eta > 0.  Row 2 is extra (B,R corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "B")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "R")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, -3),
        (-1, -2),
        (-1, +0),
        (-1, +1),
        (+0, -3),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, -2),
        (+1, +0),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bm_0 * _phi_coupled(B, b) + _psi(B) * (bj_0 * n1_0**2 - bp_0)
    M[0, 1] = bj_0 * dy * n1_0 * n2_0 / (dx * R * (R + 1))
    M[0, 2] = bm_0 * _couple_off_fwd(B, b)
    M[1, 0] = -bj_1 * dx * n1_1 * n2_1 / (dy * B * (B + 1))
    M[1, 1] = -bm_1 * -_phi(R) - _psi(R) * (bj_1 * n2_1**2 - bp_1)
    M[2, 0] = -bm_2 * _couple_off_rev(B, b)
    M[2, 2] = -bm_2 * _phi_mirror(B, b) - _psi(b) * (bj_2 * n1_2**2 - bp_2)

    d = np.zeros(3)
    d[0] = a_tau_0 * bm_0 * dy * n1_0 + a_0 * bm_0 * _couple_avg(B, b) + b_0 * dy * n2_0
    d[1] = -a_tau_1 * bm_1 * dx * n2_1 - a_1 * bm_1 * -_phi(R) + b_1 * dx * n1_1
    d[2] = a_tau_2 * bm_2 * dy * n1_2 - a_2 * bm_2 * _couple_avg(B, b) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 2] = bj_0 * dy * n1_0 * n2_0 * (B * R + B + R) / (dx * (R + 1))  # (-1, +0)
    N[0, 3] = -bj_0 * dy * n1_0 * n2_0 * B / dx  # (-1, +1)
    N[0, 6] = bm_0 * _couple_avg(B, b)  # (+0, -1)
    N[0, 7] = (
        bj_0 * dx * n1_0**2 * B * R
        + bj_0 * dx * n1_0**2 * R
        - bj_0 * dy * n1_0 * n2_0 * B**2 * R
        - bj_0 * dy * n1_0 * n2_0 * B * R
        + bj_0 * dy * n1_0 * n2_0 * B
        - bp_0 * dx * B * R
        - bp_0 * dx * R
    ) / (dx * B * R)  # (+0, +0)
    N[0, 8] = (
        -B
        * (
            bj_0 * dx * n1_0**2
            - bj_0 * dy * n1_0 * n2_0 * B
            - bj_0 * dy * n1_0 * n2_0
            - bp_0 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[1, 2] = (
        -R
        * (
            bj_1 * dx * n1_1 * n2_1 * R
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            + bp_1 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[1, 3] = bj_1 * dx * n1_1 * n2_1 * R / dy  # (-1, +1)
    N[1, 7] = (
        bj_1 * dx * n1_1 * n2_1 * B * R**2
        + bj_1 * dx * n1_1 * n2_1 * B * R
        - bj_1 * dx * n1_1 * n2_1 * R
        - bj_1 * dy * n2_1**2 * B * R
        - bj_1 * dy * n2_1**2 * B
        + bp_1 * dy * B * R
        + bp_1 * dy * B
    ) / (dy * B * R)  # (+0, +0)
    N[1, 8] = -bj_1 * dx * n1_1 * n2_1 * (B * R + B + R) / (dy * (B + 1))  # (+0, +1)
    N[1, 10] = bm_1 * (R - 2) / (R - 1)  # (+1, +0)
    N[1, 11] = -bm_1 * (R - 1) / (R - 2)  # (+2, +0)
    N[2, 0] = -bj_2 * dy * n1_2 * n2_2 * b / dx  # (-1, -3)
    N[2, 1] = (1 / 2) * bj_2 * dy * n1_2 * n2_2 * (2 * b + 1) / dx  # (-1, -2)
    N[2, 4] = (
        b
        * (
            bj_2 * dx * n1_2**2
            + bj_2 * dy * n1_2 * n2_2 * b
            + bj_2 * dy * n1_2 * n2_2
            - bp_2 * dx
        )
        / (dx * (b + 1))
    )  # (+0, -3)
    N[2, 5] = -(
        bj_2 * dx * n1_2**2 * b
        + bj_2 * dx * n1_2**2
        + bj_2 * dy * n1_2 * n2_2 * b**2
        - bp_2 * dx * b
        - bp_2 * dx
    ) / (dx * b)  # (+0, -2)
    N[2, 6] = -bm_2 * _couple_avg(B, b)  # (+0, -1)
    N[2, 9] = -1 / 2 * bj_2 * dy * n1_2 * n2_2 / dx  # (+1, -2)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc8_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 8, eta < 0.  Row 2 is extra (B,R corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "B")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "R")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, +0),
        (-1, +1),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, +0),
        (+2, -1),
        (+2, +0),
        (+2, +1),
        (+3, -1),
        (+3, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bp_0 * -_phi(B) + _psi(B) * (bj_0 * n1_0**2 + bm_0)
    M[0, 1] = bj_0 * dy * n1_0 * n2_0 / (dx * R * (R + 1))
    M[1, 0] = -bj_1 * dx * n1_1 * n2_1 / (dy * B * (B + 1))
    M[1, 1] = bp_1 * _phi_coupled(R, r) - _psi(R) * (bj_1 * n2_1**2 + bm_1)
    M[1, 2] = bp_1 * _couple_off_fwd(R, r)
    M[2, 2] = -bp_2 * _couple_off_rev(R, r)

    d = np.zeros(3)
    d[0] = a_tau_0 * bp_0 * dy * n1_0 + a_0 * bp_0 * -_phi(B) + b_0 * dy * n2_0
    d[1] = (
        -a_tau_1 * bp_1 * dx * n2_1 - a_1 * bp_1 * _couple_avg(R, r) + b_1 * dx * n1_1
    )
    d[2] = (
        -a_tau_2 * bp_2 * dx * n2_2 + a_2 * bp_2 * _couple_avg(R, r) + b_2 * dx * n1_2
    )

    N = np.zeros((3, len(offsets)))
    N[0, 0] = bj_0 * dy * n1_0 * n2_0 * (B * R + B + R) / (dx * (R + 1))  # (-1, +0)
    N[0, 1] = -bj_0 * dy * n1_0 * n2_0 * B / dx  # (-1, +1)
    N[0, 2] = -bp_0 * (B - 1) / (B - 2)  # (+0, -2)
    N[0, 3] = bp_0 * (B - 2) / (B - 1)  # (+0, -1)
    N[0, 4] = (
        bj_0 * dx * n1_0**2 * B * R
        + bj_0 * dx * n1_0**2 * R
        - bj_0 * dy * n1_0 * n2_0 * B**2 * R
        - bj_0 * dy * n1_0 * n2_0 * B * R
        + bj_0 * dy * n1_0 * n2_0 * B
        + bm_0 * dx * B * R
        + bm_0 * dx * R
    ) / (dx * B * R)  # (+0, +0)
    N[0, 5] = (
        -B
        * (
            bj_0 * dx * n1_0**2
            - bj_0 * dy * n1_0 * n2_0 * B
            - bj_0 * dy * n1_0 * n2_0
            + bm_0 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[1, 0] = (
        -R
        * (
            bj_1 * dx * n1_1 * n2_1 * R
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            - bm_1 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[1, 1] = bj_1 * dx * n1_1 * n2_1 * R / dy  # (-1, +1)
    N[1, 4] = (
        bj_1 * dx * n1_1 * n2_1 * B * R**2
        + bj_1 * dx * n1_1 * n2_1 * B * R
        - bj_1 * dx * n1_1 * n2_1 * R
        - bj_1 * dy * n2_1**2 * B * R
        - bj_1 * dy * n2_1**2 * B
        - bm_1 * dy * B * R
        - bm_1 * dy * B
    ) / (dy * B * R)  # (+0, +0)
    N[1, 5] = -bj_1 * dx * n1_1 * n2_1 * (B * R + B + R) / (dy * (B + 1))  # (+0, +1)
    N[1, 6] = bp_1 * _couple_avg(R, r)  # (+1, +0)
    N[2, 6] = -bp_2 * _couple_avg(R, r)  # (+1, +0)
    N[2, 7] = (1 / 2) * bj_2 * dx * n1_2 * n2_2 * (2 * r + 1) / dy  # (+2, -1)
    N[2, 8] = -(
        bj_2 * dx * n1_2 * n2_2 * r**2
        - bj_2 * dy * n2_2**2 * r
        - bj_2 * dy * n2_2**2
        - bm_2 * dy * r
        - bm_2 * dy
    ) / (dy * r)  # (+2, +0)
    N[2, 9] = -1 / 2 * bj_2 * dx * n1_2 * n2_2 / dy  # (+2, +1)
    N[2, 10] = -bj_2 * dx * n1_2 * n2_2 * r / dy  # (+3, -1)
    N[2, 11] = (
        r
        * (
            bj_2 * dx * n1_2 * n2_2 * r
            + bj_2 * dx * n1_2 * n2_2
            - bj_2 * dy * n2_2**2
            - bm_2 * dy
        )
        / (dy * (r + 1))
    )  # (+3, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case3_sc8_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 3 sub-case 8, eta > 0.  Row 2 is extra (B,R corner)."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "B")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "R")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "extra")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, +0),
        (-1, +1),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, +0),
        (+2, -1),
        (+2, +0),
        (+2, +1),
        (+3, -1),
        (+3, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bm_0 * -_phi(B) + _psi(B) * (bj_0 * n1_0**2 - bp_0)
    M[0, 1] = bj_0 * dy * n1_0 * n2_0 / (dx * R * (R + 1))
    M[1, 0] = -bj_1 * dx * n1_1 * n2_1 / (dy * B * (B + 1))
    M[1, 1] = -bm_1 * _phi_coupled(R, r) - _psi(R) * (bj_1 * n2_1**2 - bp_1)
    M[1, 2] = -bm_1 * _couple_off_fwd(R, r)
    M[2, 2] = bm_2 * _couple_off_rev(R, r)

    d = np.zeros(3)
    d[0] = a_tau_0 * bm_0 * dy * n1_0 + a_0 * bm_0 * -_phi(B) + b_0 * dy * n2_0
    d[1] = (
        -a_tau_1 * bm_1 * dx * n2_1 - a_1 * bm_1 * _couple_avg(R, r) + b_1 * dx * n1_1
    )
    d[2] = (
        -a_tau_2 * bm_2 * dx * n2_2 + a_2 * bm_2 * _couple_avg(R, r) + b_2 * dx * n1_2
    )

    N = np.zeros((3, len(offsets)))
    N[0, 0] = bj_0 * dy * n1_0 * n2_0 * (B * R + B + R) / (dx * (R + 1))  # (-1, +0)
    N[0, 1] = -bj_0 * dy * n1_0 * n2_0 * B / dx  # (-1, +1)
    N[0, 2] = bm_0 * (B - 1) / (B - 2)  # (+0, -2)
    N[0, 3] = -bm_0 * (B - 2) / (B - 1)  # (+0, -1)
    N[0, 4] = (
        bj_0 * dx * n1_0**2 * B * R
        + bj_0 * dx * n1_0**2 * R
        - bj_0 * dy * n1_0 * n2_0 * B**2 * R
        - bj_0 * dy * n1_0 * n2_0 * B * R
        + bj_0 * dy * n1_0 * n2_0 * B
        - bp_0 * dx * B * R
        - bp_0 * dx * R
    ) / (dx * B * R)  # (+0, +0)
    N[0, 5] = (
        -B
        * (
            bj_0 * dx * n1_0**2
            - bj_0 * dy * n1_0 * n2_0 * B
            - bj_0 * dy * n1_0 * n2_0
            - bp_0 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[1, 0] = (
        -R
        * (
            bj_1 * dx * n1_1 * n2_1 * R
            + bj_1 * dx * n1_1 * n2_1
            - bj_1 * dy * n2_1**2
            + bp_1 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[1, 1] = bj_1 * dx * n1_1 * n2_1 * R / dy  # (-1, +1)
    N[1, 4] = (
        bj_1 * dx * n1_1 * n2_1 * B * R**2
        + bj_1 * dx * n1_1 * n2_1 * B * R
        - bj_1 * dx * n1_1 * n2_1 * R
        - bj_1 * dy * n2_1**2 * B * R
        - bj_1 * dy * n2_1**2 * B
        + bp_1 * dy * B * R
        + bp_1 * dy * B
    ) / (dy * B * R)  # (+0, +0)
    N[1, 5] = -bj_1 * dx * n1_1 * n2_1 * (B * R + B + R) / (dy * (B + 1))  # (+0, +1)
    N[1, 6] = -bm_1 * _couple_avg(R, r)  # (+1, +0)
    N[2, 6] = bm_2 * _couple_avg(R, r)  # (+1, +0)
    N[2, 7] = (1 / 2) * bj_2 * dx * n1_2 * n2_2 * (2 * r + 1) / dy  # (+2, -1)
    N[2, 8] = -(
        bj_2 * dx * n1_2 * n2_2 * r**2
        - bj_2 * dy * n2_2**2 * r
        - bj_2 * dy * n2_2**2
        + bp_2 * dy * r
        + bp_2 * dy
    ) / (dy * r)  # (+2, +0)
    N[2, 9] = -1 / 2 * bj_2 * dx * n1_2 * n2_2 / dy  # (+2, +1)
    N[2, 10] = -bj_2 * dx * n1_2 * n2_2 * r / dy  # (+3, -1)
    N[2, 11] = (
        r
        * (
            bj_2 * dx * n1_2 * n2_2 * r
            + bj_2 * dx * n1_2 * n2_2
            - bj_2 * dy * n2_2**2
            + bp_2 * dy
        )
        / (dy * (r + 1))
    )  # (+3, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case4_sc1_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 4 sub-case 1, eta < 0."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "T")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "L")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, -1),
        (-1, +0),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, +0),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = (
        bj_0
        * n2_0
        * (dx * n1_0 * L * R - dx * n1_0 * R - dy * n2_0 * L - 2 * dy * n2_0 * R)
        / (dy * R * (L + R))
        - bm_0 * (L + 2 * R) / (R * (L + R))
        + bp_0 * -_phi(R)
    )
    M[0, 1] = bj_0 * dx * n1_0 * n2_0 / (dy * T * (T + 1))
    M[0, 2] = -bj_0 * n2_0 * R * (dx * n1_0 * R + dx * n1_0 + dy * n2_0) / (
        dy * L * (L + R)
    ) - bm_0 * R / (L * (L + R))
    M[1, 0] = bj_1 * dy * n1_1 * n2_1 * (L * T + L - T) / (dx * R * (L + R))
    M[1, 1] = bp_1 * -_phi(T) - _psi(T) * (bj_1 * n1_1**2 + bm_1)
    M[1, 2] = -bj_1 * dy * n1_1 * n2_1 * (R * T + R + T) / (dx * L * (L + R))
    M[2, 0] = -bj_2 * n2_2 * L * (dx * n1_2 * L - dx * n1_2 - dy * n2_2) / (
        dy * R * (L + R)
    ) + bm_2 * L / (R * (L + R))
    M[2, 1] = bj_2 * dx * n1_2 * n2_2 / (dy * T * (T + 1))
    M[2, 2] = (
        bj_2
        * n2_2
        * (dx * n1_2 * L * R + dx * n1_2 * L + 2 * dy * n2_2 * L + dy * n2_2 * R)
        / (dy * L * (L + R))
        + bm_2 * (2 * L + R) / (L * (L + R))
        - bp_2 * -_phi(L)
    )

    d = np.zeros(3)
    d[0] = -a_tau_0 * bp_0 * dx * n2_0 - a_0 * bp_0 * -_phi(R) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bp_1 * dy * n1_1 - a_1 * bp_1 * -_phi(T) + b_1 * dy * n2_1
    d[2] = -a_tau_2 * bp_2 * dx * n2_2 + a_2 * bp_2 * -_phi(L) + b_2 * dx * n1_2

    N = np.zeros((3, len(offsets)))
    N[0, 1] = -bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, -1)
    N[0, 3] = bj_0 * dx * n1_0 * n2_0 * (R * T + R + T) / (dy * (T + 1))  # (+0, -1)
    N[0, 4] = (
        bj_0 * dx * n1_0 * n2_0 * L * R
        - bj_0 * dx * n1_0 * n2_0 * R**2 * T
        - bj_0 * dx * n1_0 * n2_0 * R * T
        - bj_0 * dy * n2_0**2 * L * T
        - bj_0 * dy * n2_0**2 * R * T
        - bm_0 * dy * L * T
        - bm_0 * dy * R * T
    ) / (dy * L * R * T)  # (+0, +0)
    N[0, 7] = -bp_0 * (R - 2) / (R - 1)  # (+1, +0)
    N[0, 8] = bp_0 * (R - 1) / (R - 2)  # (+2, +0)
    N[1, 1] = -bj_1 * dy * n1_1 * n2_1 * T / dx  # (-1, -1)
    N[1, 3] = (
        T
        * (
            bj_1 * dx * n1_1**2
            + bj_1 * dy * n1_1 * n2_1 * T
            + bj_1 * dy * n1_1 * n2_1
            + bm_1 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[1, 4] = -(
        bj_1 * dx * n1_1**2 * L * R * T
        + bj_1 * dx * n1_1**2 * L * R
        - bj_1 * dy * n1_1 * n2_1 * L * T**2
        - bj_1 * dy * n1_1 * n2_1 * L * T
        + bj_1 * dy * n1_1 * n2_1 * R * T**2
        + bj_1 * dy * n1_1 * n2_1 * R * T
        + bj_1 * dy * n1_1 * n2_1 * T**2
        + bm_1 * dx * L * R * T
        + bm_1 * dx * L * R
    ) / (dx * L * R * T)  # (+0, +0)
    N[1, 5] = -bp_1 * (T - 2) / (T - 1)  # (+0, +1)
    N[1, 6] = bp_1 * (T - 1) / (T - 2)  # (+0, +2)
    N[2, 0] = -bp_2 * (L - 1) / (L - 2)  # (-2, +0)
    N[2, 1] = bj_2 * dx * n1_2 * n2_2 * L / dy  # (-1, -1)
    N[2, 2] = bp_2 * (L - 2) / (L - 1)  # (-1, +0)
    N[2, 3] = -bj_2 * dx * n1_2 * n2_2 * (L * T + L - T) / (dy * (T + 1))  # (+0, -1)
    N[2, 4] = -(
        bj_2 * dx * n1_2 * n2_2 * L**2 * T
        - bj_2 * dx * n1_2 * n2_2 * L * R
        - bj_2 * dx * n1_2 * n2_2 * L * T
        - bj_2 * dy * n2_2**2 * L * T
        - bj_2 * dy * n2_2**2 * R * T
        - bm_2 * dy * L * T
        - bm_2 * dy * R * T
    ) / (dy * L * R * T)  # (+0, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case4_sc1_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 4 sub-case 1, eta > 0."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "T")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "L")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, -1),
        (-1, +0),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, +0),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = (
        bj_0
        * n2_0
        * (dx * n1_0 * L * R - dx * n1_0 * R - dy * n2_0 * L - 2 * dy * n2_0 * R)
        / (dy * R * (L + R))
        - bm_0 * -_phi(R)
        + bp_0 * (L + 2 * R) / (R * (L + R))
    )
    M[0, 1] = bj_0 * dx * n1_0 * n2_0 / (dy * T * (T + 1))
    M[0, 2] = -bj_0 * n2_0 * R * (dx * n1_0 * R + dx * n1_0 + dy * n2_0) / (
        dy * L * (L + R)
    ) + bp_0 * R / (L * (L + R))
    M[1, 0] = bj_1 * dy * n1_1 * n2_1 * (L * T + L - T) / (dx * R * (L + R))
    M[1, 1] = -bm_1 * -_phi(T) - _psi(T) * (bj_1 * n1_1**2 - bp_1)
    M[1, 2] = -bj_1 * dy * n1_1 * n2_1 * (R * T + R + T) / (dx * L * (L + R))
    M[2, 0] = -bj_2 * n2_2 * L * (dx * n1_2 * L - dx * n1_2 - dy * n2_2) / (
        dy * R * (L + R)
    ) - bp_2 * L / (R * (L + R))
    M[2, 1] = bj_2 * dx * n1_2 * n2_2 / (dy * T * (T + 1))
    M[2, 2] = (
        bj_2
        * n2_2
        * (dx * n1_2 * L * R + dx * n1_2 * L + 2 * dy * n2_2 * L + dy * n2_2 * R)
        / (dy * L * (L + R))
        + bm_2 * -_phi(L)
        - bp_2 * (2 * L + R) / (L * (L + R))
    )

    d = np.zeros(3)
    d[0] = -a_tau_0 * bm_0 * dx * n2_0 - a_0 * bm_0 * -_phi(R) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bm_1 * dy * n1_1 - a_1 * bm_1 * -_phi(T) + b_1 * dy * n2_1
    d[2] = -a_tau_2 * bm_2 * dx * n2_2 + a_2 * bm_2 * -_phi(L) + b_2 * dx * n1_2

    N = np.zeros((3, len(offsets)))
    N[0, 1] = -bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, -1)
    N[0, 3] = bj_0 * dx * n1_0 * n2_0 * (R * T + R + T) / (dy * (T + 1))  # (+0, -1)
    N[0, 4] = (
        bj_0 * dx * n1_0 * n2_0 * L * R
        - bj_0 * dx * n1_0 * n2_0 * R**2 * T
        - bj_0 * dx * n1_0 * n2_0 * R * T
        - bj_0 * dy * n2_0**2 * L * T
        - bj_0 * dy * n2_0**2 * R * T
        + bp_0 * dy * L * T
        + bp_0 * dy * R * T
    ) / (dy * L * R * T)  # (+0, +0)
    N[0, 7] = bm_0 * (R - 2) / (R - 1)  # (+1, +0)
    N[0, 8] = -bm_0 * (R - 1) / (R - 2)  # (+2, +0)
    N[1, 1] = -bj_1 * dy * n1_1 * n2_1 * T / dx  # (-1, -1)
    N[1, 3] = (
        T
        * (
            bj_1 * dx * n1_1**2
            + bj_1 * dy * n1_1 * n2_1 * T
            + bj_1 * dy * n1_1 * n2_1
            - bp_1 * dx
        )
        / (dx * (T + 1))
    )  # (+0, -1)
    N[1, 4] = -(
        bj_1 * dx * n1_1**2 * L * R * T
        + bj_1 * dx * n1_1**2 * L * R
        - bj_1 * dy * n1_1 * n2_1 * L * T**2
        - bj_1 * dy * n1_1 * n2_1 * L * T
        + bj_1 * dy * n1_1 * n2_1 * R * T**2
        + bj_1 * dy * n1_1 * n2_1 * R * T
        + bj_1 * dy * n1_1 * n2_1 * T**2
        - bp_1 * dx * L * R * T
        - bp_1 * dx * L * R
    ) / (dx * L * R * T)  # (+0, +0)
    N[1, 5] = bm_1 * (T - 2) / (T - 1)  # (+0, +1)
    N[1, 6] = -bm_1 * (T - 1) / (T - 2)  # (+0, +2)
    N[2, 0] = bm_2 * (L - 1) / (L - 2)  # (-2, +0)
    N[2, 1] = bj_2 * dx * n1_2 * n2_2 * L / dy  # (-1, -1)
    N[2, 2] = -bm_2 * (L - 2) / (L - 1)  # (-1, +0)
    N[2, 3] = -bj_2 * dx * n1_2 * n2_2 * (L * T + L - T) / (dy * (T + 1))  # (+0, -1)
    N[2, 4] = -(
        bj_2 * dx * n1_2 * n2_2 * L**2 * T
        - bj_2 * dx * n1_2 * n2_2 * L * R
        - bj_2 * dx * n1_2 * n2_2 * L * T
        - bj_2 * dy * n2_2**2 * L * T
        - bj_2 * dy * n2_2**2 * R * T
        + bp_2 * dy * L * T
        + bp_2 * dy * R * T
    ) / (dy * L * R * T)  # (+0, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case4_sc2_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 4 sub-case 2, eta < 0."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "T")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "B")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, +0),
        (-1, +1),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, +0),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = bp_0 * -_phi(R) - _psi(R) * (bj_0 * n2_0**2 + bm_0)
    M[0, 1] = bj_0 * dx * n1_0 * n2_0 * (B * R + B + R) / (dy * T * (B + T))
    M[0, 2] = -bj_0 * dx * n1_0 * n2_0 * (R * T - R + T) / (dy * B * (B + T))
    M[1, 0] = bj_1 * dy * n1_1 * n2_1 / (dx * R * (R + 1))
    M[1, 1] = (
        -bj_1
        * n1_1
        * (dx * n1_1 * B + 2 * dx * n1_1 * T - dy * n2_1 * B * T - dy * n2_1 * T)
        / (dx * T * (B + T))
        - bm_1 * (B + 2 * T) / (T * (B + T))
        + bp_1 * -_phi(T)
    )
    M[1, 2] = -bj_1 * n1_1 * T * (dx * n1_1 + dy * n2_1 * T - dy * n2_1) / (
        dx * B * (B + T)
    ) - bm_1 * T / (B * (B + T))
    M[2, 0] = bj_2 * dy * n1_2 * n2_2 / (dx * R * (R + 1))
    M[2, 1] = bj_2 * n1_2 * B * (dx * n1_2 - dy * n2_2 * B - dy * n2_2) / (
        dx * T * (B + T)
    ) + bm_2 * B / (T * (B + T))
    M[2, 2] = (
        bj_2
        * n1_2
        * (2 * dx * n1_2 * B + dx * n1_2 * T + dy * n2_2 * B * T - dy * n2_2 * B)
        / (dx * B * (B + T))
        + bm_2 * (2 * B + T) / (B * (B + T))
        - bp_2 * -_phi(B)
    )

    d = np.zeros(3)
    d[0] = -a_tau_0 * bp_0 * dx * n2_0 - a_0 * bp_0 * -_phi(R) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bp_1 * dy * n1_1 - a_1 * bp_1 * -_phi(T) + b_1 * dy * n2_1
    d[2] = a_tau_2 * bp_2 * dy * n1_2 + a_2 * bp_2 * -_phi(B) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 0] = (
        -R
        * (
            bj_0 * dx * n1_0 * n2_0 * R
            + bj_0 * dx * n1_0 * n2_0
            - bj_0 * dy * n2_0**2
            - bm_0 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[0, 1] = bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, +1)
    N[0, 4] = (
        bj_0 * dx * n1_0 * n2_0 * B * R**2
        + bj_0 * dx * n1_0 * n2_0 * B * R
        - bj_0 * dx * n1_0 * n2_0 * R**2 * T
        + bj_0 * dx * n1_0 * n2_0 * R**2
        - bj_0 * dx * n1_0 * n2_0 * R * T
        - bj_0 * dy * n2_0**2 * B * R * T
        - bj_0 * dy * n2_0**2 * B * T
        - bm_0 * dy * B * R * T
        - bm_0 * dy * B * T
    ) / (dy * B * R * T)  # (+0, +0)
    N[0, 7] = -bp_0 * (R - 2) / (R - 1)  # (+1, +0)
    N[0, 8] = bp_0 * (R - 1) / (R - 2)  # (+2, +0)
    N[1, 0] = -bj_1 * dy * n1_1 * n2_1 * (R * T - R + T) / (dx * (R + 1))  # (-1, +0)
    N[1, 1] = bj_1 * dy * n1_1 * n2_1 * T / dx  # (-1, +1)
    N[1, 4] = -(
        bj_1 * dx * n1_1**2 * B * R
        + bj_1 * dx * n1_1**2 * R * T
        - bj_1 * dy * n1_1 * n2_1 * B * T
        + bj_1 * dy * n1_1 * n2_1 * R * T**2
        - bj_1 * dy * n1_1 * n2_1 * R * T
        + bm_1 * dx * B * R
        + bm_1 * dx * R * T
    ) / (dx * B * R * T)  # (+0, +0)
    N[1, 5] = -bp_1 * (T - 2) / (T - 1)  # (+0, +1)
    N[1, 6] = bp_1 * (T - 1) / (T - 2)  # (+0, +2)
    N[2, 0] = bj_2 * dy * n1_2 * n2_2 * (B * R + B + R) / (dx * (R + 1))  # (-1, +0)
    N[2, 1] = -bj_2 * dy * n1_2 * n2_2 * B / dx  # (-1, +1)
    N[2, 2] = -bp_2 * (B - 1) / (B - 2)  # (+0, -2)
    N[2, 3] = bp_2 * (B - 2) / (B - 1)  # (+0, -1)
    N[2, 4] = (
        bj_2 * dx * n1_2**2 * B * R
        + bj_2 * dx * n1_2**2 * R * T
        - bj_2 * dy * n1_2 * n2_2 * B**2 * R
        - bj_2 * dy * n1_2 * n2_2 * B * R
        + bj_2 * dy * n1_2 * n2_2 * B * T
        + bm_2 * dx * B * R
        + bm_2 * dx * R * T
    ) / (dx * B * R * T)  # (+0, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case4_sc2_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 4 sub-case 2, eta > 0."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "T")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "B")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-1, +0),
        (-1, +1),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, +0),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = -bm_0 * -_phi(R) - _psi(R) * (bj_0 * n2_0**2 - bp_0)
    M[0, 1] = bj_0 * dx * n1_0 * n2_0 * (B * R + B + R) / (dy * T * (B + T))
    M[0, 2] = -bj_0 * dx * n1_0 * n2_0 * (R * T - R + T) / (dy * B * (B + T))
    M[1, 0] = bj_1 * dy * n1_1 * n2_1 / (dx * R * (R + 1))
    M[1, 1] = (
        -bj_1
        * n1_1
        * (dx * n1_1 * B + 2 * dx * n1_1 * T - dy * n2_1 * B * T - dy * n2_1 * T)
        / (dx * T * (B + T))
        - bm_1 * -_phi(T)
        + bp_1 * (B + 2 * T) / (T * (B + T))
    )
    M[1, 2] = -bj_1 * n1_1 * T * (dx * n1_1 + dy * n2_1 * T - dy * n2_1) / (
        dx * B * (B + T)
    ) + bp_1 * T / (B * (B + T))
    M[2, 0] = bj_2 * dy * n1_2 * n2_2 / (dx * R * (R + 1))
    M[2, 1] = bj_2 * n1_2 * B * (dx * n1_2 - dy * n2_2 * B - dy * n2_2) / (
        dx * T * (B + T)
    ) - bp_2 * B / (T * (B + T))
    M[2, 2] = (
        bj_2
        * n1_2
        * (2 * dx * n1_2 * B + dx * n1_2 * T + dy * n2_2 * B * T - dy * n2_2 * B)
        / (dx * B * (B + T))
        + bm_2 * -_phi(B)
        - bp_2 * (2 * B + T) / (B * (B + T))
    )

    d = np.zeros(3)
    d[0] = -a_tau_0 * bm_0 * dx * n2_0 - a_0 * bm_0 * -_phi(R) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bm_1 * dy * n1_1 - a_1 * bm_1 * -_phi(T) + b_1 * dy * n2_1
    d[2] = a_tau_2 * bm_2 * dy * n1_2 + a_2 * bm_2 * -_phi(B) + b_2 * dy * n2_2

    N = np.zeros((3, len(offsets)))
    N[0, 0] = (
        -R
        * (
            bj_0 * dx * n1_0 * n2_0 * R
            + bj_0 * dx * n1_0 * n2_0
            - bj_0 * dy * n2_0**2
            + bp_0 * dy
        )
        / (dy * (R + 1))
    )  # (-1, +0)
    N[0, 1] = bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, +1)
    N[0, 4] = (
        bj_0 * dx * n1_0 * n2_0 * B * R**2
        + bj_0 * dx * n1_0 * n2_0 * B * R
        - bj_0 * dx * n1_0 * n2_0 * R**2 * T
        + bj_0 * dx * n1_0 * n2_0 * R**2
        - bj_0 * dx * n1_0 * n2_0 * R * T
        - bj_0 * dy * n2_0**2 * B * R * T
        - bj_0 * dy * n2_0**2 * B * T
        + bp_0 * dy * B * R * T
        + bp_0 * dy * B * T
    ) / (dy * B * R * T)  # (+0, +0)
    N[0, 7] = bm_0 * (R - 2) / (R - 1)  # (+1, +0)
    N[0, 8] = -bm_0 * (R - 1) / (R - 2)  # (+2, +0)
    N[1, 0] = -bj_1 * dy * n1_1 * n2_1 * (R * T - R + T) / (dx * (R + 1))  # (-1, +0)
    N[1, 1] = bj_1 * dy * n1_1 * n2_1 * T / dx  # (-1, +1)
    N[1, 4] = -(
        bj_1 * dx * n1_1**2 * B * R
        + bj_1 * dx * n1_1**2 * R * T
        - bj_1 * dy * n1_1 * n2_1 * B * T
        + bj_1 * dy * n1_1 * n2_1 * R * T**2
        - bj_1 * dy * n1_1 * n2_1 * R * T
        - bp_1 * dx * B * R
        - bp_1 * dx * R * T
    ) / (dx * B * R * T)  # (+0, +0)
    N[1, 5] = bm_1 * (T - 2) / (T - 1)  # (+0, +1)
    N[1, 6] = -bm_1 * (T - 1) / (T - 2)  # (+0, +2)
    N[2, 0] = bj_2 * dy * n1_2 * n2_2 * (B * R + B + R) / (dx * (R + 1))  # (-1, +0)
    N[2, 1] = -bj_2 * dy * n1_2 * n2_2 * B / dx  # (-1, +1)
    N[2, 2] = bm_2 * (B - 1) / (B - 2)  # (+0, -2)
    N[2, 3] = -bm_2 * (B - 2) / (B - 1)  # (+0, -1)
    N[2, 4] = (
        bj_2 * dx * n1_2**2 * B * R
        + bj_2 * dx * n1_2**2 * R * T
        - bj_2 * dy * n1_2 * n2_2 * B**2 * R
        - bj_2 * dy * n1_2 * n2_2 * B * R
        + bj_2 * dy * n1_2 * n2_2 * B * T
        - bp_2 * dx * B * R
        - bp_2 * dx * R * T
    ) / (dx * B * R * T)  # (+0, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case4_sc3_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 4 sub-case 3, eta < 0."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "B")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "L")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, +0),
        (-1, +1),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, +0),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = (
        -bj_0
        * n2_0
        * (dx * n1_0 * L * R - dx * n1_0 * R + dy * n2_0 * L + 2 * dy * n2_0 * R)
        / (dy * R * (L + R))
        - bm_0 * (L + 2 * R) / (R * (L + R))
        + bp_0 * -_phi(R)
    )
    M[0, 1] = -bj_0 * dx * n1_0 * n2_0 / (dy * B * (B + 1))
    M[0, 2] = bj_0 * n2_0 * R * (dx * n1_0 * R + dx * n1_0 - dy * n2_0) / (
        dy * L * (L + R)
    ) - bm_0 * R / (L * (L + R))
    M[1, 0] = bj_1 * dy * n1_1 * n2_1 * (B * L - B + L) / (dx * R * (L + R))
    M[1, 1] = -bp_1 * -_phi(B) + _psi(B) * (bj_1 * n1_1**2 + bm_1)
    M[1, 2] = -bj_1 * dy * n1_1 * n2_1 * (B * R + B + R) / (dx * L * (L + R))
    M[2, 0] = bj_2 * n2_2 * L * (dx * n1_2 * L - dx * n1_2 + dy * n2_2) / (
        dy * R * (L + R)
    ) + bm_2 * L / (R * (L + R))
    M[2, 1] = -bj_2 * dx * n1_2 * n2_2 / (dy * B * (B + 1))
    M[2, 2] = (
        -bj_2
        * n2_2
        * (dx * n1_2 * L * R + dx * n1_2 * L - 2 * dy * n2_2 * L - dy * n2_2 * R)
        / (dy * L * (L + R))
        + bm_2 * (2 * L + R) / (L * (L + R))
        - bp_2 * -_phi(L)
    )

    d = np.zeros(3)
    d[0] = -a_tau_0 * bp_0 * dx * n2_0 - a_0 * bp_0 * -_phi(R) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bp_1 * dy * n1_1 + a_1 * bp_1 * -_phi(B) + b_1 * dy * n2_1
    d[2] = -a_tau_2 * bp_2 * dx * n2_2 + a_2 * bp_2 * -_phi(L) + b_2 * dx * n1_2

    N = np.zeros((3, len(offsets)))
    N[0, 2] = bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, +1)
    N[0, 5] = (
        bj_0 * dx * n1_0 * n2_0 * B * R**2
        + bj_0 * dx * n1_0 * n2_0 * B * R
        - bj_0 * dx * n1_0 * n2_0 * L * R
        - bj_0 * dy * n2_0**2 * B * L
        - bj_0 * dy * n2_0**2 * B * R
        - bm_0 * dy * B * L
        - bm_0 * dy * B * R
    ) / (dy * B * L * R)  # (+0, +0)
    N[0, 6] = -bj_0 * dx * n1_0 * n2_0 * (B * R + B + R) / (dy * (B + 1))  # (+0, +1)
    N[0, 7] = -bp_0 * (R - 2) / (R - 1)  # (+1, +0)
    N[0, 8] = bp_0 * (R - 1) / (R - 2)  # (+2, +0)
    N[1, 2] = -bj_1 * dy * n1_1 * n2_1 * B / dx  # (-1, +1)
    N[1, 3] = -bp_1 * (B - 1) / (B - 2)  # (+0, -2)
    N[1, 4] = bp_1 * (B - 2) / (B - 1)  # (+0, -1)
    N[1, 5] = (
        bj_1 * dx * n1_1**2 * B * L * R
        + bj_1 * dx * n1_1**2 * L * R
        + bj_1 * dy * n1_1 * n2_1 * B**2 * L
        - bj_1 * dy * n1_1 * n2_1 * B**2 * R
        - bj_1 * dy * n1_1 * n2_1 * B**2
        + bj_1 * dy * n1_1 * n2_1 * B * L
        - bj_1 * dy * n1_1 * n2_1 * B * R
        + bm_1 * dx * B * L * R
        + bm_1 * dx * L * R
    ) / (dx * B * L * R)  # (+0, +0)
    N[1, 6] = (
        -B
        * (
            bj_1 * dx * n1_1**2
            - bj_1 * dy * n1_1 * n2_1 * B
            - bj_1 * dy * n1_1 * n2_1
            + bm_1 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[2, 0] = -bp_2 * (L - 1) / (L - 2)  # (-2, +0)
    N[2, 1] = bp_2 * (L - 2) / (L - 1)  # (-1, +0)
    N[2, 2] = -bj_2 * dx * n1_2 * n2_2 * L / dy  # (-1, +1)
    N[2, 5] = (
        bj_2 * dx * n1_2 * n2_2 * B * L**2
        - bj_2 * dx * n1_2 * n2_2 * B * L
        - bj_2 * dx * n1_2 * n2_2 * L * R
        + bj_2 * dy * n2_2**2 * B * L
        + bj_2 * dy * n2_2**2 * B * R
        + bm_2 * dy * B * L
        + bm_2 * dy * B * R
    ) / (dy * B * L * R)  # (+0, +0)
    N[2, 6] = bj_2 * dx * n1_2 * n2_2 * (B * L - B + L) / (dy * (B + 1))  # (+0, +1)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case4_sc3_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 4 sub-case 3, eta > 0."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "R")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "B")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "L")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, +0),
        (-1, +1),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+1, +0),
        (+2, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = (
        -bj_0
        * n2_0
        * (dx * n1_0 * L * R - dx * n1_0 * R + dy * n2_0 * L + 2 * dy * n2_0 * R)
        / (dy * R * (L + R))
        - bm_0 * -_phi(R)
        + bp_0 * (L + 2 * R) / (R * (L + R))
    )
    M[0, 1] = -bj_0 * dx * n1_0 * n2_0 / (dy * B * (B + 1))
    M[0, 2] = bj_0 * n2_0 * R * (dx * n1_0 * R + dx * n1_0 - dy * n2_0) / (
        dy * L * (L + R)
    ) + bp_0 * R / (L * (L + R))
    M[1, 0] = bj_1 * dy * n1_1 * n2_1 * (B * L - B + L) / (dx * R * (L + R))
    M[1, 1] = bm_1 * -_phi(B) + _psi(B) * (bj_1 * n1_1**2 - bp_1)
    M[1, 2] = -bj_1 * dy * n1_1 * n2_1 * (B * R + B + R) / (dx * L * (L + R))
    M[2, 0] = bj_2 * n2_2 * L * (dx * n1_2 * L - dx * n1_2 + dy * n2_2) / (
        dy * R * (L + R)
    ) - bp_2 * L / (R * (L + R))
    M[2, 1] = -bj_2 * dx * n1_2 * n2_2 / (dy * B * (B + 1))
    M[2, 2] = (
        -bj_2
        * n2_2
        * (dx * n1_2 * L * R + dx * n1_2 * L - 2 * dy * n2_2 * L - dy * n2_2 * R)
        / (dy * L * (L + R))
        + bm_2 * -_phi(L)
        - bp_2 * (2 * L + R) / (L * (L + R))
    )

    d = np.zeros(3)
    d[0] = -a_tau_0 * bm_0 * dx * n2_0 - a_0 * bm_0 * -_phi(R) + b_0 * dx * n1_0
    d[1] = a_tau_1 * bm_1 * dy * n1_1 + a_1 * bm_1 * -_phi(B) + b_1 * dy * n2_1
    d[2] = -a_tau_2 * bm_2 * dx * n2_2 + a_2 * bm_2 * -_phi(L) + b_2 * dx * n1_2

    N = np.zeros((3, len(offsets)))
    N[0, 2] = bj_0 * dx * n1_0 * n2_0 * R / dy  # (-1, +1)
    N[0, 5] = (
        bj_0 * dx * n1_0 * n2_0 * B * R**2
        + bj_0 * dx * n1_0 * n2_0 * B * R
        - bj_0 * dx * n1_0 * n2_0 * L * R
        - bj_0 * dy * n2_0**2 * B * L
        - bj_0 * dy * n2_0**2 * B * R
        + bp_0 * dy * B * L
        + bp_0 * dy * B * R
    ) / (dy * B * L * R)  # (+0, +0)
    N[0, 6] = -bj_0 * dx * n1_0 * n2_0 * (B * R + B + R) / (dy * (B + 1))  # (+0, +1)
    N[0, 7] = bm_0 * (R - 2) / (R - 1)  # (+1, +0)
    N[0, 8] = -bm_0 * (R - 1) / (R - 2)  # (+2, +0)
    N[1, 2] = -bj_1 * dy * n1_1 * n2_1 * B / dx  # (-1, +1)
    N[1, 3] = bm_1 * (B - 1) / (B - 2)  # (+0, -2)
    N[1, 4] = -bm_1 * (B - 2) / (B - 1)  # (+0, -1)
    N[1, 5] = (
        bj_1 * dx * n1_1**2 * B * L * R
        + bj_1 * dx * n1_1**2 * L * R
        + bj_1 * dy * n1_1 * n2_1 * B**2 * L
        - bj_1 * dy * n1_1 * n2_1 * B**2 * R
        - bj_1 * dy * n1_1 * n2_1 * B**2
        + bj_1 * dy * n1_1 * n2_1 * B * L
        - bj_1 * dy * n1_1 * n2_1 * B * R
        - bp_1 * dx * B * L * R
        - bp_1 * dx * L * R
    ) / (dx * B * L * R)  # (+0, +0)
    N[1, 6] = (
        -B
        * (
            bj_1 * dx * n1_1**2
            - bj_1 * dy * n1_1 * n2_1 * B
            - bj_1 * dy * n1_1 * n2_1
            - bp_1 * dx
        )
        / (dx * (B + 1))
    )  # (+0, +1)
    N[2, 0] = bm_2 * (L - 1) / (L - 2)  # (-2, +0)
    N[2, 1] = -bm_2 * (L - 2) / (L - 1)  # (-1, +0)
    N[2, 2] = -bj_2 * dx * n1_2 * n2_2 * L / dy  # (-1, +1)
    N[2, 5] = (
        bj_2 * dx * n1_2 * n2_2 * B * L**2
        - bj_2 * dx * n1_2 * n2_2 * B * L
        - bj_2 * dx * n1_2 * n2_2 * L * R
        + bj_2 * dy * n2_2**2 * B * L
        + bj_2 * dy * n2_2**2 * B * R
        - bp_2 * dy * B * L
        - bp_2 * dy * B * R
    ) / (dy * B * L * R)  # (+0, +0)
    N[2, 6] = bj_2 * dx * n1_2 * n2_2 * (B * L - B + L) / (dy * (B + 1))  # (+0, +1)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case4_sc4_eta_m1(geom, theta_inputs, betas_per_row):
    """Case 4 sub-case 4, eta < 0."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "T")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "B")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "L")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, +0),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, -1),
        (+1, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = (
        -bj_0
        * n1_0
        * (dx * n1_0 * B + 2 * dx * n1_0 * T + dy * n2_0 * B * T - dy * n2_0 * T)
        / (dx * T * (B + T))
        - bm_0 * (B + 2 * T) / (T * (B + T))
        + bp_0 * -_phi(T)
    )
    M[0, 1] = -bj_0 * n1_0 * T * (dx * n1_0 - dy * n2_0 * T - dy * n2_0) / (
        dx * B * (B + T)
    ) - bm_0 * T / (B * (B + T))
    M[0, 2] = -bj_0 * dy * n1_0 * n2_0 / (dx * L * (L + 1))
    M[1, 0] = bj_1 * n1_1 * B * (dx * n1_1 + dy * n2_1 * B - dy * n2_1) / (
        dx * T * (B + T)
    ) + bm_1 * B / (T * (B + T))
    M[1, 1] = (
        bj_1
        * n1_1
        * (2 * dx * n1_1 * B + dx * n1_1 * T - dy * n2_1 * B * T - dy * n2_1 * B)
        / (dx * B * (B + T))
        + bm_1 * (2 * B + T) / (B * (B + T))
        - bp_1 * -_phi(B)
    )
    M[1, 2] = -bj_1 * dy * n1_1 * n2_1 / (dx * L * (L + 1))
    M[2, 0] = bj_2 * dx * n1_2 * n2_2 * (B * L + B - L) / (dy * T * (B + T))
    M[2, 1] = -bj_2 * dx * n1_2 * n2_2 * (L * T + L + T) / (dy * B * (B + T))
    M[2, 2] = -bp_2 * -_phi(L) + _psi(L) * (bj_2 * n2_2**2 + bm_2)

    d = np.zeros(3)
    d[0] = a_tau_0 * bp_0 * dy * n1_0 - a_0 * bp_0 * -_phi(T) + b_0 * dy * n2_0
    d[1] = a_tau_1 * bp_1 * dy * n1_1 + a_1 * bp_1 * -_phi(B) + b_1 * dy * n2_1
    d[2] = -a_tau_2 * bp_2 * dx * n2_2 + a_2 * bp_2 * -_phi(L) + b_2 * dx * n1_2

    N = np.zeros((3, len(offsets)))
    N[0, 4] = -(
        bj_0 * dx * n1_0**2 * B * L
        + bj_0 * dx * n1_0**2 * L * T
        + bj_0 * dy * n1_0 * n2_0 * B * T
        - bj_0 * dy * n1_0 * n2_0 * L * T**2
        - bj_0 * dy * n1_0 * n2_0 * L * T
        + bm_0 * dx * B * L
        + bm_0 * dx * L * T
    ) / (dx * B * L * T)  # (+0, +0)
    N[0, 5] = -bp_0 * (T - 2) / (T - 1)  # (+0, +1)
    N[0, 6] = bp_0 * (T - 1) / (T - 2)  # (+0, +2)
    N[0, 7] = bj_0 * dy * n1_0 * n2_0 * T / dx  # (+1, -1)
    N[0, 8] = -bj_0 * dy * n1_0 * n2_0 * (L * T + L + T) / (dx * (L + 1))  # (+1, +0)
    N[1, 2] = -bp_1 * (B - 1) / (B - 2)  # (+0, -2)
    N[1, 3] = bp_1 * (B - 2) / (B - 1)  # (+0, -1)
    N[1, 4] = (
        bj_1 * dx * n1_1**2 * B * L
        + bj_1 * dx * n1_1**2 * L * T
        + bj_1 * dy * n1_1 * n2_1 * B**2 * L
        - bj_1 * dy * n1_1 * n2_1 * B * L
        - bj_1 * dy * n1_1 * n2_1 * B * T
        + bm_1 * dx * B * L
        + bm_1 * dx * L * T
    ) / (dx * B * L * T)  # (+0, +0)
    N[1, 7] = -bj_1 * dy * n1_1 * n2_1 * B / dx  # (+1, -1)
    N[1, 8] = bj_1 * dy * n1_1 * n2_1 * (B * L + B - L) / (dx * (L + 1))  # (+1, +0)
    N[2, 0] = -bp_2 * (L - 1) / (L - 2)  # (-2, +0)
    N[2, 1] = bp_2 * (L - 2) / (L - 1)  # (-1, +0)
    N[2, 4] = (
        bj_2 * dx * n1_2 * n2_2 * B * L**2
        + bj_2 * dx * n1_2 * n2_2 * B * L
        - bj_2 * dx * n1_2 * n2_2 * L**2 * T
        - bj_2 * dx * n1_2 * n2_2 * L**2
        - bj_2 * dx * n1_2 * n2_2 * L * T
        + bj_2 * dy * n2_2**2 * B * L * T
        + bj_2 * dy * n2_2**2 * B * T
        + bm_2 * dy * B * L * T
        + bm_2 * dy * B * T
    ) / (dy * B * L * T)  # (+0, +0)
    N[2, 7] = -bj_2 * dx * n1_2 * n2_2 * L / dy  # (+1, -1)
    N[2, 8] = (
        L
        * (
            bj_2 * dx * n1_2 * n2_2 * L
            + bj_2 * dx * n1_2 * n2_2
            - bj_2 * dy * n2_2**2
            - bm_2 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


def _case4_sc4_eta_p1(geom, theta_inputs, betas_per_row):
    """Case 4 sub-case 4, eta > 0."""
    R, T, L, B = (
        theta_inputs["R"],
        theta_inputs["T"],
        theta_inputs["L"],
        theta_inputs["B"],
    )
    r, t, l, b = (
        theta_inputs["r"],
        theta_inputs["t"],
        theta_inputs["l"],
        theta_inputs["b"],
    )

    n1_0, n2_0, a_0, b_0, a_tau_0 = _row_geom_data(geom, "T")
    n1_1, n2_1, a_1, b_1, a_tau_1 = _row_geom_data(geom, "B")
    n1_2, n2_2, a_2, b_2, a_tau_2 = _row_geom_data(geom, "L")
    bp_0, bm_0, bj_0 = betas_per_row[0]
    bp_1, bm_1, bj_1 = betas_per_row[1]
    bp_2, bm_2, bj_2 = betas_per_row[2]

    offsets = [
        (-2, +0),
        (-1, +0),
        (+0, -2),
        (+0, -1),
        (+0, +0),
        (+0, +1),
        (+0, +2),
        (+1, -1),
        (+1, +0),
    ]

    M = np.zeros((3, 3))
    M[0, 0] = (
        -bj_0
        * n1_0
        * (dx * n1_0 * B + 2 * dx * n1_0 * T + dy * n2_0 * B * T - dy * n2_0 * T)
        / (dx * T * (B + T))
        - bm_0 * -_phi(T)
        + bp_0 * (B + 2 * T) / (T * (B + T))
    )
    M[0, 1] = -bj_0 * n1_0 * T * (dx * n1_0 - dy * n2_0 * T - dy * n2_0) / (
        dx * B * (B + T)
    ) + bp_0 * T / (B * (B + T))
    M[0, 2] = -bj_0 * dy * n1_0 * n2_0 / (dx * L * (L + 1))
    M[1, 0] = bj_1 * n1_1 * B * (dx * n1_1 + dy * n2_1 * B - dy * n2_1) / (
        dx * T * (B + T)
    ) - bp_1 * B / (T * (B + T))
    M[1, 1] = (
        bj_1
        * n1_1
        * (2 * dx * n1_1 * B + dx * n1_1 * T - dy * n2_1 * B * T - dy * n2_1 * B)
        / (dx * B * (B + T))
        + bm_1 * -_phi(B)
        - bp_1 * (2 * B + T) / (B * (B + T))
    )
    M[1, 2] = -bj_1 * dy * n1_1 * n2_1 / (dx * L * (L + 1))
    M[2, 0] = bj_2 * dx * n1_2 * n2_2 * (B * L + B - L) / (dy * T * (B + T))
    M[2, 1] = -bj_2 * dx * n1_2 * n2_2 * (L * T + L + T) / (dy * B * (B + T))
    M[2, 2] = bm_2 * -_phi(L) + _psi(L) * (bj_2 * n2_2**2 - bp_2)

    d = np.zeros(3)
    d[0] = a_tau_0 * bm_0 * dy * n1_0 - a_0 * bm_0 * -_phi(T) + b_0 * dy * n2_0
    d[1] = a_tau_1 * bm_1 * dy * n1_1 + a_1 * bm_1 * -_phi(B) + b_1 * dy * n2_1
    d[2] = -a_tau_2 * bm_2 * dx * n2_2 + a_2 * bm_2 * -_phi(L) + b_2 * dx * n1_2

    N = np.zeros((3, len(offsets)))
    N[0, 4] = -(
        bj_0 * dx * n1_0**2 * B * L
        + bj_0 * dx * n1_0**2 * L * T
        + bj_0 * dy * n1_0 * n2_0 * B * T
        - bj_0 * dy * n1_0 * n2_0 * L * T**2
        - bj_0 * dy * n1_0 * n2_0 * L * T
        - bp_0 * dx * B * L
        - bp_0 * dx * L * T
    ) / (dx * B * L * T)  # (+0, +0)
    N[0, 5] = bm_0 * (T - 2) / (T - 1)  # (+0, +1)
    N[0, 6] = -bm_0 * (T - 1) / (T - 2)  # (+0, +2)
    N[0, 7] = bj_0 * dy * n1_0 * n2_0 * T / dx  # (+1, -1)
    N[0, 8] = -bj_0 * dy * n1_0 * n2_0 * (L * T + L + T) / (dx * (L + 1))  # (+1, +0)
    N[1, 2] = bm_1 * (B - 1) / (B - 2)  # (+0, -2)
    N[1, 3] = -bm_1 * (B - 2) / (B - 1)  # (+0, -1)
    N[1, 4] = (
        bj_1 * dx * n1_1**2 * B * L
        + bj_1 * dx * n1_1**2 * L * T
        + bj_1 * dy * n1_1 * n2_1 * B**2 * L
        - bj_1 * dy * n1_1 * n2_1 * B * L
        - bj_1 * dy * n1_1 * n2_1 * B * T
        - bp_1 * dx * B * L
        - bp_1 * dx * L * T
    ) / (dx * B * L * T)  # (+0, +0)
    N[1, 7] = -bj_1 * dy * n1_1 * n2_1 * B / dx  # (+1, -1)
    N[1, 8] = bj_1 * dy * n1_1 * n2_1 * (B * L + B - L) / (dx * (L + 1))  # (+1, +0)
    N[2, 0] = bm_2 * (L - 1) / (L - 2)  # (-2, +0)
    N[2, 1] = -bm_2 * (L - 2) / (L - 1)  # (-1, +0)
    N[2, 4] = (
        bj_2 * dx * n1_2 * n2_2 * B * L**2
        + bj_2 * dx * n1_2 * n2_2 * B * L
        - bj_2 * dx * n1_2 * n2_2 * L**2 * T
        - bj_2 * dx * n1_2 * n2_2 * L**2
        - bj_2 * dx * n1_2 * n2_2 * L * T
        + bj_2 * dy * n2_2**2 * B * L * T
        + bj_2 * dy * n2_2**2 * B * T
        - bp_2 * dy * B * L * T
        - bp_2 * dy * B * T
    ) / (dy * B * L * T)  # (+0, +0)
    N[2, 7] = -bj_2 * dx * n1_2 * n2_2 * L / dy  # (+1, -1)
    N[2, 8] = (
        L
        * (
            bj_2 * dx * n1_2 * n2_2 * L
            + bj_2 * dx * n1_2 * n2_2
            - bj_2 * dy * n2_2**2
            + bp_2 * dy
        )
        / (dy * (L + 1))
    )  # (+1, +0)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, offsets


# Row interfaces for each sub-case (needed by _row_betas)
CASE3_ROW_IFACES = {
    (1, -1): ("R", "extra", "T"),
    (1, 1): ("R", "extra", "T"),
    (2, -1): ("R", "T", "extra"),
    (2, 1): ("R", "T", "extra"),
    (3, -1): ("T", "L", "extra"),
    (3, 1): ("T", "L", "extra"),
    (4, -1): ("T", "L", "extra"),
    (4, 1): ("T", "L", "extra"),
    (5, -1): ("L", "B", "extra"),
    (5, 1): ("L", "B", "extra"),
    (6, -1): ("L", "B", "extra"),
    (6, 1): ("L", "B", "extra"),
    (7, -1): ("B", "R", "extra"),
    (7, 1): ("B", "R", "extra"),
    (8, -1): ("B", "R", "extra"),
    (8, 1): ("B", "R", "extra"),
}
CASE4_ROW_IFACES = {
    (1, -1): ("R", "T", "L"),
    (1, 1): ("R", "T", "L"),
    (2, -1): ("R", "T", "B"),
    (2, 1): ("R", "T", "B"),
    (3, -1): ("R", "B", "L"),
    (3, 1): ("R", "B", "L"),
    (4, -1): ("T", "B", "L"),
    (4, 1): ("T", "B", "L"),
}

# Dispatch dicts
CASE3_FUNCS = {}
CASE4_FUNCS = {}
CASE3_FUNCS[(1, -1)] = _case3_sc1_eta_m1
CASE3_FUNCS[(1, 1)] = _case3_sc1_eta_p1
CASE3_FUNCS[(2, -1)] = _case3_sc2_eta_m1
CASE3_FUNCS[(2, 1)] = _case3_sc2_eta_p1
CASE3_FUNCS[(3, -1)] = _case3_sc3_eta_m1
CASE3_FUNCS[(3, 1)] = _case3_sc3_eta_p1
CASE3_FUNCS[(4, -1)] = _case3_sc4_eta_m1
CASE3_FUNCS[(4, 1)] = _case3_sc4_eta_p1
CASE3_FUNCS[(5, -1)] = _case3_sc5_eta_m1
CASE3_FUNCS[(5, 1)] = _case3_sc5_eta_p1
CASE3_FUNCS[(6, -1)] = _case3_sc6_eta_m1
CASE3_FUNCS[(6, 1)] = _case3_sc6_eta_p1
CASE3_FUNCS[(7, -1)] = _case3_sc7_eta_m1
CASE3_FUNCS[(7, 1)] = _case3_sc7_eta_p1
CASE3_FUNCS[(8, -1)] = _case3_sc8_eta_m1
CASE3_FUNCS[(8, 1)] = _case3_sc8_eta_p1
CASE4_FUNCS[(1, -1)] = _case4_sc1_eta_m1
CASE4_FUNCS[(1, 1)] = _case4_sc1_eta_p1
CASE4_FUNCS[(2, -1)] = _case4_sc2_eta_m1
CASE4_FUNCS[(2, 1)] = _case4_sc2_eta_p1
CASE4_FUNCS[(3, -1)] = _case4_sc3_eta_m1
CASE4_FUNCS[(3, 1)] = _case4_sc3_eta_p1
CASE4_FUNCS[(4, -1)] = _case4_sc4_eta_m1
CASE4_FUNCS[(4, 1)] = _case4_sc4_eta_p1


def _assemble_case_n(
    i,
    j,
    sw_idx,
    M_inv_d,
    M_inv_N,
    all_offsets,
    eps_r,
    eps_l,
    eps_t,
    eps_b,
    theta_R,
    theta_T,
    theta_L,
    theta_B,
    bot_x,
    bot_y,
):
    """Shortley-Weller assembly for a case-3 / case-4 cell.

    `sw_idx` maps face label ('R'/'T'/'L'/'B') to the row index of the
    corresponding interface ghost in M_inv_d / M_inv_N. Faces NOT in sw_idx
    are uncut and contribute the regular grid value at the neighbour.
    """
    row_idx = index(i, j)

    # Per-face Shortley-Weller weights: (eps, theta, bot_dim, neighbour offset).
    face_data = {
        "R": (eps_r, theta_R, bot_x, (1, 0)),
        "T": (eps_t, theta_T, bot_y, (0, 1)),
        "L": (eps_l, theta_L, bot_x, (-1, 0)),
        "B": (eps_b, theta_B, bot_y, (0, -1)),
    }

    # f update from the d term of each cut face.
    for face, idx_in_M in sw_idx.items():
        eps_face, theta_face, bot_dim, _ = face_data[face]
        f[i, j] -= M_inv_d[idx_in_M] * eps_face / theta_face / bot_dim

    coefs = {}

    # Contribution to each grid point from the local 3x3 ghost solve.
    for k, off in enumerate(all_offsets):
        v = 0.0
        for face, idx_in_M in sw_idx.items():
            eps_face, theta_face, bot_dim, _ = face_data[face]
            v += M_inv_N[idx_in_M, k] * eps_face / theta_face / bot_dim
        coefs[off] = v

    # Diagonal Shortley-Weller term for u_{i,j}.
    coefs[(0, 0)] = coefs.get((0, 0), 0.0) + (
        -(eps_r / theta_R + eps_l / theta_L) / bot_x
        - (eps_t / theta_T + eps_b / theta_B) / bot_y
    )

    # Uncut neighbours pick up the regular flux term directly.
    for face, (eps_face, theta_face, bot_dim, neighbour) in face_data.items():
        if face not in sw_idx:
            coefs[neighbour] = (
                coefs.get(neighbour, 0.0) + eps_face / theta_face / bot_dim
            )

    for (di, dj), val in coefs.items():
        rows.append(row_idx)
        cols.append(index(i + di, j + dj))
        vals.append(val)


def construct_matrix():
    for i in range(nx):
        for j in range(ny):
            # if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
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
                    if extra.bit_count() == 0:
                        coeff_case2(direction, i, j)
                    elif extra.bit_count() == 1:
                        coeff_case3(direction, extra, i, j)
                    else:
                        raise ValueError(
                            f"Invalid extra direction {extra} at ({i},{j}), use finer grid."
                        )
                case 3:
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
    cd = _CD[direction]
    x, y = center(i, j)
    eta = surface(x, y)
    theta = compute_theta(direction, i, j)
    a_tau_I = interp(direction, theta, i, j, a_tau)
    a_I = interp(direction, theta, i, j, a)
    b_I = interp(direction, theta, i, j, b)
    n1_I = interp(direction, theta, i, j, n1)
    n2_I = interp(direction, theta, i, j, n2)

    theta_l, theta_r, theta_t, theta_b = cd.theta_assign(theta)

    px, py, p_axis = cd.probe_loc(x, y, theta)
    _eps_p, _eps_m, eps_jump, eps_p, eps_m = _sample_beta_legacy(px, py, p_axis, eta)

    n_t = [n1_I, n2_I][cd.n_tang]
    n_n = [n1_I, n2_I][cd.n_norm]
    d_self = dx if cd.is_x else dy
    d_other = dy if cd.is_x else dx

    d = (
        (-1 if cd.is_x else 1) * a_tau_I * eps_p * n_t * d_self
        + b_I * n_n * d_self
        + cd.sign * a_I * eps_p * _phi(theta)
    )

    if eta > 0:
        eps_p, eps_m = -_eps_m, -_eps_p

    M = -cd.sign * (
        eps_p * _phi(theta) + eps_m * _psi(theta) + eps_jump * n_t**2 * _psi(theta)
    )

    N = [
        -cd.sign * (eps_jump * n_t**2 + eps_m) * (1 + theta) / theta
        - eps_jump * n1_I * n2_I * theta * d_self / d_other,
        -cd.sign * eps_p * (theta - 2) / (theta - 1),
        cd.sign * eps_p * (theta - 1) / (theta - 2),
        cd.sign * (eps_jump * n_t**2 + eps_m) * theta / (1 + theta)
        + eps_jump * n1_I * n2_I * theta * d_self / d_other,
        eps_jump * n1_I * n2_I * (2 * theta + 1) * d_self / (2 * d_other),
        -eps_jump * n1_I * n2_I * d_self / (2 * d_other),
        -eps_jump * n1_I * n2_I * theta * d_self / d_other,
    ]
    u_arr = [u[i + di, j + dj] for (di, dj) in cd.offsets]
    u_I = (np.dot(N, u_arr) + d) / M

    # Return (u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t)
    # where the cut face's value is the ghost value u_I.
    slot_names = ["l", "r", "t", "b"]
    face_vals = {"l": u[i - 1, j], "r": u[i + 1, j], "b": u[i, j - 1], "t": u[i, j + 1]}
    face_vals[slot_names[cd.slot]] = u_I
    return (
        face_vals["l"],
        face_vals["r"],
        face_vals["b"],
        face_vals["t"],
        theta_l,
        theta_r,
        theta_b,
        theta_t,
    )


def interface_value_case2(
    direction: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute the interface value of u at the cut."""
    M_inv_d, M_inv_N, all_offsets, sw_idx, geom = _solve_case2_local(direction, i, j)
    u_arr = np.array([u[i + di, j + dj] for (di, dj) in all_offsets])
    ghosts = M_inv_N @ u_arr + M_inv_d
    return _pack_iface_values(sw_idx, ghosts, geom, i, j, u)


def interface_value_case3(
    direction: int, extra: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Reconstruct interface values (u_l, u_r, u_b, u_t) for a case-3 cell.

    Mirrors interface_value_case2 but for the three-interface stencil. Two of
    the returned interface u-values come from the local 3x3 solve; the third
    (corresponding to the side without a corner cut) is just a regular grid
    value.
    """
    x_, y_ = center(i, j)
    geom = _case3_geometry(direction, extra, i, j)

    sub_case_table = {
        (Direction.R | Direction.T, Direction.R): 1,
        (Direction.R | Direction.T, Direction.T): 2,
        (Direction.L | Direction.T, Direction.T): 3,
        (Direction.L | Direction.T, Direction.L): 4,
        (Direction.L | Direction.B, Direction.L): 5,
        (Direction.L | Direction.B, Direction.B): 6,
        (Direction.R | Direction.B, Direction.B): 7,
        (Direction.R | Direction.B, Direction.R): 8,
    }
    sub_case = sub_case_table.get((int(direction), int(extra)))
    if sub_case is None:
        raise ValueError(
            f"Invalid (direction, extra) for case 3: ({direction!r}, {extra!r})"
        )

    eta = geom["eta"]
    M_inv_d, M_inv_N, all_offsets = _solve_case3_local(
        sub_case, eta, direction, extra, geom
    )

    # Apply M_inv_N @ u_arr + M_inv_d to get the 3 interface ghost values.
    u_arr = np.array([u[i + di, j + dj] for (di, dj) in all_offsets])
    ghosts = M_inv_N @ u_arr + M_inv_d

    sw_idx = _CASE3_SW_INDEX[sub_case]
    return _pack_iface_values(sw_idx, ghosts, geom, i, j, u)


def _pack_iface_values(sw_idx, ghosts, geom, i, j, u):
    """Return (u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t).

    Cut directions take their value from `ghosts[sw_idx[face]]`; uncut
    directions return the regular grid neighbour and theta = 1.
    """
    u_l = ghosts[sw_idx["L"]] if "L" in sw_idx else u[i - 1, j]
    u_r = ghosts[sw_idx["R"]] if "R" in sw_idx else u[i + 1, j]
    u_b = ghosts[sw_idx["B"]] if "B" in sw_idx else u[i, j - 1]
    u_t = ghosts[sw_idx["T"]] if "T" in sw_idx else u[i, j + 1]
    theta_l = geom["theta_L"] if "L" in sw_idx else 1.0
    theta_r = geom["theta_R"] if "R" in sw_idx else 1.0
    theta_b = geom["theta_B"] if "B" in sw_idx else 1.0
    theta_t = geom["theta_T"] if "T" in sw_idx else 1.0
    return u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t


def interface_value_case4(
    direction: int, i: int, j: int, u: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Reconstruct interface values (u_l, u_r, u_b, u_t) for a case-4 cell.

    Mirrors interface_value_case3: three of the returned slots come from the
    local 3x3 solve, the fourth (uncut side) is just a regular grid value.
    """
    geom = _case4_geometry(direction, i, j)

    sub_case_table = {
        int(Direction.R | Direction.T | Direction.L): 1,
        int(Direction.R | Direction.T | Direction.B): 2,
        int(Direction.R | Direction.B | Direction.L): 3,
        int(Direction.T | Direction.B | Direction.L): 4,
    }
    sub_case = sub_case_table.get(int(direction))
    if sub_case is None:
        raise ValueError(f"Invalid direction for case 4: {direction!r}")

    eta = geom["eta"]
    M_inv_d, M_inv_N, all_offsets = _solve_case4_local(sub_case, eta, direction, geom)
    u_arr = np.array([u[i + di, j + dj] for (di, dj) in all_offsets])
    ghosts = M_inv_N @ u_arr + M_inv_d
    sw_idx = _CASE4_SW_INDEX[sub_case]
    return _pack_iface_values(sw_idx, ghosts, geom, i, j, u)


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
    for idx, n in enumerate(n_range):
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


if __name__ == "__main__":
    # convergence_test1()
    # convergence_test2()
    # convergence_test3()
    convergence_test4()
    # convergence_test5()

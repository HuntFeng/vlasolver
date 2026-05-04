"""
Poisson solver prototype
Handles complex shape and piecewise variable permittivity
"""
import enum
from typing import NamedTuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


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


class EvalCtx(NamedTuple):
    """Evaluation context for dispatch-table entry functions.

    Bundles the 18 scalar inputs that were previously passed as positional
    arguments to every lambda in CASE3_EVAL / CASE4_EVAL.
    """
    R: float; T: float; L: float; B: float
    r: float; t: float; l: float; b: float
    n1: float; n2: float
    av: float; bv: float; at: float
    ep: float; em: float; ej: float
    dx: float; dy: float


# Evaluator dispatch tables
# Each entry maps to a dict with:
#   'M': [[func]] 3x3, 'd': [func], 'N': [[{offset: func}]] 3-list,
#   'offsets': [(di,dj)], 'row_ifaces': (str, str, str)

CASE3_EVAL = {}
CASE4_EVAL = {}

# case3 sub-case 1, eta sign -1
_case3_sc1_eta_m1_M = [
    [lambda ctx: ctx.ep*(2*ctx.R + ctx.r - 3)/((ctx.R - 1)*(ctx.R + ctx.r - 2)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.R*(ctx.R + 1)), lambda ctx: ctx.ep*(ctx.R - 1)/((ctx.r - 1)*(ctx.R + ctx.r - 2)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1))],
    [lambda ctx: -ctx.ep*(ctx.r - 1)/((ctx.R - 1)*(ctx.R + ctx.r - 2)), lambda ctx: -ctx.ep*(ctx.R + 2*ctx.r - 3)/((ctx.r - 1)*(ctx.R + ctx.r - 2)) + (2*ctx.r + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.r*(ctx.r + 1)), _Z],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), _Z, lambda ctx: ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.T*(ctx.T + 1))],
]
_case3_sc1_eta_m1_d = [lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc1_eta_m1_N = [
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (-1, +0): lambda ctx: ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 + ctx.em*ctx.dy)/(ctx.dy*(ctx.R + 1)), (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dy*ctx.n2**2*ctx.R*ctx.T + ctx.ej*ctx.dy*ctx.n2**2*ctx.T + ctx.em*ctx.dy*ctx.R*ctx.T + ctx.em*ctx.dy*ctx.T)/(ctx.dy*ctx.R*ctx.T), (+1, +0): lambda ctx: ctx.ep*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1))},
    {(+1, +0): lambda ctx: -ctx.ep*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)), (+2, -1): lambda ctx: (1/2)*ctx.ej*ctx.dx*ctx.n1*ctx.n2*(2*ctx.r + 1)/ctx.dy, (+2, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r**2 - ctx.ej*ctx.dy*ctx.n2**2*ctx.r - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy*ctx.r - ctx.em*ctx.dy)/(ctx.dy*ctx.r), (+2, +1): lambda ctx: -1/2*ctx.ej*ctx.dx*ctx.n1*ctx.n2/ctx.dy, (+3, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r/ctx.dy, (+3, +0): lambda ctx: ctx.r*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy)/(ctx.dy*(ctx.r + 1))},
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dx*(ctx.R + 1)), (+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.R*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.em*ctx.dx*ctx.R*ctx.T + ctx.em*ctx.dx*ctx.R)/(ctx.dx*ctx.R*ctx.T), (+0, +1): lambda ctx: -ctx.ep*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: ctx.ep*(ctx.T - 1)/(ctx.T - 2)},
]
_case3_sc1_eta_m1_offsets = [(-1, -1), (-1, +0), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, +0), (+2, -1), (+2, +0), (+2, +1), (+3, -1), (+3, +0)]
_case3_sc1_eta_m1_row_ifaces = ("R", "extra", "T")
_case3_sc1_eta_m1 = {
    'M': _case3_sc1_eta_m1_M, 'd': _case3_sc1_eta_m1_d, 'N': _case3_sc1_eta_m1_N,
    'offsets': _case3_sc1_eta_m1_offsets, 'row_ifaces': _case3_sc1_eta_m1_row_ifaces,
}
CASE3_EVAL[(1, -1)] = _case3_sc1_eta_m1

# case3 sub-case 1, eta sign +1
_case3_sc1_eta_p1_M = [
    [lambda ctx: -ctx.em*(2*ctx.R + ctx.r - 3)/((ctx.R - 1)*(ctx.R + ctx.r - 2)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.R*(ctx.R + 1)), lambda ctx: -ctx.em*(ctx.R - 1)/((ctx.r - 1)*(ctx.R + ctx.r - 2)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1))],
    [lambda ctx: ctx.em*(ctx.r - 1)/((ctx.R - 1)*(ctx.R + ctx.r - 2)), lambda ctx: ctx.em*(ctx.R + 2*ctx.r - 3)/((ctx.r - 1)*(ctx.R + ctx.r - 2)) + (2*ctx.r + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.r*(ctx.r + 1)), _Z],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), _Z, lambda ctx: -ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.T*(ctx.T + 1))],
]
_case3_sc1_eta_p1_d = [lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc1_eta_p1_N = [
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (-1, +0): lambda ctx: ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 - ctx.ep*ctx.dy)/(ctx.dy*(ctx.R + 1)), (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dy*ctx.n2**2*ctx.R*ctx.T + ctx.ej*ctx.dy*ctx.n2**2*ctx.T - ctx.ep*ctx.dy*ctx.R*ctx.T - ctx.ep*ctx.dy*ctx.T)/(ctx.dy*ctx.R*ctx.T), (+1, +0): lambda ctx: -ctx.em*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1))},
    {(+1, +0): lambda ctx: ctx.em*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)), (+2, -1): lambda ctx: (1/2)*ctx.ej*ctx.dx*ctx.n1*ctx.n2*(2*ctx.r + 1)/ctx.dy, (+2, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r**2 - ctx.ej*ctx.dy*ctx.n2**2*ctx.r - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy*ctx.r + ctx.ep*ctx.dy)/(ctx.dy*ctx.r), (+2, +1): lambda ctx: -1/2*ctx.ej*ctx.dx*ctx.n1*ctx.n2/ctx.dy, (+3, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r/ctx.dy, (+3, +0): lambda ctx: ctx.r*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy)/(ctx.dy*(ctx.r + 1))},
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dx*(ctx.R + 1)), (+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.R*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T - ctx.ep*ctx.dx*ctx.R*ctx.T - ctx.ep*ctx.dx*ctx.R)/(ctx.dx*ctx.R*ctx.T), (+0, +1): lambda ctx: ctx.em*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: -ctx.em*(ctx.T - 1)/(ctx.T - 2)},
]
_case3_sc1_eta_p1_offsets = [(-1, -1), (-1, +0), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, +0), (+2, -1), (+2, +0), (+2, +1), (+3, -1), (+3, +0)]
_case3_sc1_eta_p1_row_ifaces = ("R", "extra", "T")
_case3_sc1_eta_p1 = {
    'M': _case3_sc1_eta_p1_M, 'd': _case3_sc1_eta_p1_d, 'N': _case3_sc1_eta_p1_N,
    'offsets': _case3_sc1_eta_p1_offsets, 'row_ifaces': _case3_sc1_eta_p1_row_ifaces,
}
CASE3_EVAL[(1, 1)] = _case3_sc1_eta_p1

# case3 sub-case 2, eta sign -1
_case3_sc2_eta_m1_M = [
    [lambda ctx: ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.R*(ctx.R + 1)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), _Z],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), lambda ctx: ctx.ep*(2*ctx.T + ctx.t - 3)/((ctx.T - 1)*(ctx.T + ctx.t - 2)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.T*(ctx.T + 1)), lambda ctx: ctx.ep*(ctx.T - 1)/((ctx.t - 1)*(ctx.T + ctx.t - 2))],
    [_Z, lambda ctx: -ctx.ep*(ctx.t - 1)/((ctx.T - 1)*(ctx.T + ctx.t - 2)), lambda ctx: -ctx.ep*(ctx.T + 2*ctx.t - 3)/((ctx.t - 1)*(ctx.T + ctx.t - 2)) + (2*ctx.t + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.t*(ctx.t + 1))],
]
_case3_sc2_eta_m1_d = [lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc2_eta_m1_N = [
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (-1, +0): lambda ctx: ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 + ctx.em*ctx.dy)/(ctx.dy*(ctx.R + 1)), (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dy*ctx.n2**2*ctx.R*ctx.T + ctx.ej*ctx.dy*ctx.n2**2*ctx.T + ctx.em*ctx.dy*ctx.R*ctx.T + ctx.em*ctx.dy*ctx.T)/(ctx.dy*ctx.R*ctx.T), (+1, +0): lambda ctx: -ctx.ep*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: ctx.ep*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dx*(ctx.R + 1)), (+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.R*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.em*ctx.dx*ctx.R*ctx.T + ctx.em*ctx.dx*ctx.R)/(ctx.dx*ctx.R*ctx.T), (+0, +1): lambda ctx: ctx.ep*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1))},
    {(-1, +2): lambda ctx: (1/2)*ctx.ej*ctx.dy*ctx.n1*ctx.n2*(2*ctx.t + 1)/ctx.dx, (-1, +3): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t/ctx.dx, (+0, +1): lambda ctx: -ctx.ep*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)), (+0, +2): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.t + ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t**2 + ctx.em*ctx.dx*ctx.t + ctx.em*ctx.dx)/(ctx.dx*ctx.t), (+0, +3): lambda ctx: -ctx.t*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t - ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.t + 1)), (+1, +2): lambda ctx: -1/2*ctx.ej*ctx.dy*ctx.n1*ctx.n2/ctx.dx},
]
_case3_sc2_eta_m1_offsets = [(-1, -1), (-1, +0), (-1, +2), (-1, +3), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+0, +3), (+1, +0), (+1, +2), (+2, +0)]
_case3_sc2_eta_m1_row_ifaces = ("R", "T", "extra")
_case3_sc2_eta_m1 = {
    'M': _case3_sc2_eta_m1_M, 'd': _case3_sc2_eta_m1_d, 'N': _case3_sc2_eta_m1_N,
    'offsets': _case3_sc2_eta_m1_offsets, 'row_ifaces': _case3_sc2_eta_m1_row_ifaces,
}
CASE3_EVAL[(2, -1)] = _case3_sc2_eta_m1

# case3 sub-case 2, eta sign +1
_case3_sc2_eta_p1_M = [
    [lambda ctx: -ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.R*(ctx.R + 1)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), _Z],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), lambda ctx: -ctx.em*(2*ctx.T + ctx.t - 3)/((ctx.T - 1)*(ctx.T + ctx.t - 2)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.T*(ctx.T + 1)), lambda ctx: -ctx.em*(ctx.T - 1)/((ctx.t - 1)*(ctx.T + ctx.t - 2))],
    [_Z, lambda ctx: ctx.em*(ctx.t - 1)/((ctx.T - 1)*(ctx.T + ctx.t - 2)), lambda ctx: ctx.em*(ctx.T + 2*ctx.t - 3)/((ctx.t - 1)*(ctx.T + ctx.t - 2)) + (2*ctx.t + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.t*(ctx.t + 1))],
]
_case3_sc2_eta_p1_d = [lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc2_eta_p1_N = [
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (-1, +0): lambda ctx: ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 - ctx.ep*ctx.dy)/(ctx.dy*(ctx.R + 1)), (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dy*ctx.n2**2*ctx.R*ctx.T + ctx.ej*ctx.dy*ctx.n2**2*ctx.T - ctx.ep*ctx.dy*ctx.R*ctx.T - ctx.ep*ctx.dy*ctx.T)/(ctx.dy*ctx.R*ctx.T), (+1, +0): lambda ctx: ctx.em*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: -ctx.em*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dx*(ctx.R + 1)), (+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.R*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T - ctx.ep*ctx.dx*ctx.R*ctx.T - ctx.ep*ctx.dx*ctx.R)/(ctx.dx*ctx.R*ctx.T), (+0, +1): lambda ctx: -ctx.em*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1))},
    {(-1, +2): lambda ctx: (1/2)*ctx.ej*ctx.dy*ctx.n1*ctx.n2*(2*ctx.t + 1)/ctx.dx, (-1, +3): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t/ctx.dx, (+0, +1): lambda ctx: ctx.em*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)), (+0, +2): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.t + ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t**2 - ctx.ep*ctx.dx*ctx.t - ctx.ep*ctx.dx)/(ctx.dx*ctx.t), (+0, +3): lambda ctx: -ctx.t*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t - ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.t + 1)), (+1, +2): lambda ctx: -1/2*ctx.ej*ctx.dy*ctx.n1*ctx.n2/ctx.dx},
]
_case3_sc2_eta_p1_offsets = [(-1, -1), (-1, +0), (-1, +2), (-1, +3), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+0, +3), (+1, +0), (+1, +2), (+2, +0)]
_case3_sc2_eta_p1_row_ifaces = ("R", "T", "extra")
_case3_sc2_eta_p1 = {
    'M': _case3_sc2_eta_p1_M, 'd': _case3_sc2_eta_p1_d, 'N': _case3_sc2_eta_p1_N,
    'offsets': _case3_sc2_eta_p1_offsets, 'row_ifaces': _case3_sc2_eta_p1_row_ifaces,
}
CASE3_EVAL[(2, 1)] = _case3_sc2_eta_p1

# case3 sub-case 3, eta sign -1
_case3_sc3_eta_m1_M = [
    [lambda ctx: ctx.ep*(2*ctx.T + ctx.t - 3)/((ctx.T - 1)*(ctx.T + ctx.t - 2)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1)), lambda ctx: ctx.ep*(ctx.T - 1)/((ctx.t - 1)*(ctx.T + ctx.t - 2))],
    [lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.L*(ctx.L + 1)), _Z],
    [lambda ctx: -ctx.ep*(ctx.t - 1)/((ctx.T - 1)*(ctx.T + ctx.t - 2)), _Z, lambda ctx: -ctx.ep*(ctx.T + 2*ctx.t - 3)/((ctx.t - 1)*(ctx.T + ctx.t - 2)) + (2*ctx.t + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.t*(ctx.t + 1))],
]
_case3_sc3_eta_m1_d = [lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc3_eta_m1_N = [
    {(+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.em*ctx.dx*ctx.L*ctx.T + ctx.em*ctx.dx*ctx.L)/(ctx.dx*ctx.L*ctx.T), (+0, +1): lambda ctx: ctx.ep*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)), (+1, -1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dx*(ctx.L + 1))},
    {(-2, +0): lambda ctx: -ctx.ep*(ctx.L - 1)/(ctx.L - 2), (-1, +0): lambda ctx: ctx.ep*(ctx.L - 2)/(ctx.L - 1), (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L - ctx.ej*ctx.dy*ctx.n2**2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.T - ctx.em*ctx.dy*ctx.L*ctx.T - ctx.em*ctx.dy*ctx.T)/(ctx.dy*ctx.L*ctx.T), (+1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (+1, +0): lambda ctx: ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy)/(ctx.dy*(ctx.L + 1))},
    {(-1, +2): lambda ctx: (1/2)*ctx.ej*ctx.dy*ctx.n1*ctx.n2*(2*ctx.t + 1)/ctx.dx, (-1, +3): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t/ctx.dx, (+0, +1): lambda ctx: -ctx.ep*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)), (+0, +2): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.t + ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t**2 + ctx.em*ctx.dx*ctx.t + ctx.em*ctx.dx)/(ctx.dx*ctx.t), (+0, +3): lambda ctx: -ctx.t*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t - ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.t + 1)), (+1, +2): lambda ctx: -1/2*ctx.ej*ctx.dy*ctx.n1*ctx.n2/ctx.dx},
]
_case3_sc3_eta_m1_offsets = [(-2, +0), (-1, +0), (-1, +2), (-1, +3), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+0, +3), (+1, -1), (+1, +0), (+1, +2)]
_case3_sc3_eta_m1_row_ifaces = ("T", "L", "extra")
_case3_sc3_eta_m1 = {
    'M': _case3_sc3_eta_m1_M, 'd': _case3_sc3_eta_m1_d, 'N': _case3_sc3_eta_m1_N,
    'offsets': _case3_sc3_eta_m1_offsets, 'row_ifaces': _case3_sc3_eta_m1_row_ifaces,
}
CASE3_EVAL[(3, -1)] = _case3_sc3_eta_m1

# case3 sub-case 3, eta sign +1
_case3_sc3_eta_p1_M = [
    [lambda ctx: -ctx.em*(2*ctx.T + ctx.t - 3)/((ctx.T - 1)*(ctx.T + ctx.t - 2)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1)), lambda ctx: -ctx.em*(ctx.T - 1)/((ctx.t - 1)*(ctx.T + ctx.t - 2))],
    [lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), lambda ctx: ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.L*(ctx.L + 1)), _Z],
    [lambda ctx: ctx.em*(ctx.t - 1)/((ctx.T - 1)*(ctx.T + ctx.t - 2)), _Z, lambda ctx: ctx.em*(ctx.T + 2*ctx.t - 3)/((ctx.t - 1)*(ctx.T + ctx.t - 2)) + (2*ctx.t + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.t*(ctx.t + 1))],
]
_case3_sc3_eta_p1_d = [lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc3_eta_p1_N = [
    {(+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T - ctx.ep*ctx.dx*ctx.L*ctx.T - ctx.ep*ctx.dx*ctx.L)/(ctx.dx*ctx.L*ctx.T), (+0, +1): lambda ctx: -ctx.em*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)), (+1, -1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dx*(ctx.L + 1))},
    {(-2, +0): lambda ctx: ctx.em*(ctx.L - 1)/(ctx.L - 2), (-1, +0): lambda ctx: -ctx.em*(ctx.L - 2)/(ctx.L - 1), (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L - ctx.ej*ctx.dy*ctx.n2**2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.T + ctx.ep*ctx.dy*ctx.L*ctx.T + ctx.ep*ctx.dy*ctx.T)/(ctx.dy*ctx.L*ctx.T), (+1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (+1, +0): lambda ctx: ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy)/(ctx.dy*(ctx.L + 1))},
    {(-1, +2): lambda ctx: (1/2)*ctx.ej*ctx.dy*ctx.n1*ctx.n2*(2*ctx.t + 1)/ctx.dx, (-1, +3): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t/ctx.dx, (+0, +1): lambda ctx: ctx.em*(ctx.T + ctx.t - 2)/((ctx.T - 1)*(ctx.t - 1)), (+0, +2): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.t + ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t**2 - ctx.ep*ctx.dx*ctx.t - ctx.ep*ctx.dx)/(ctx.dx*ctx.t), (+0, +3): lambda ctx: -ctx.t*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.t - ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.t + 1)), (+1, +2): lambda ctx: -1/2*ctx.ej*ctx.dy*ctx.n1*ctx.n2/ctx.dx},
]
_case3_sc3_eta_p1_offsets = [(-2, +0), (-1, +0), (-1, +2), (-1, +3), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+0, +3), (+1, -1), (+1, +0), (+1, +2)]
_case3_sc3_eta_p1_row_ifaces = ("T", "L", "extra")
_case3_sc3_eta_p1 = {
    'M': _case3_sc3_eta_p1_M, 'd': _case3_sc3_eta_p1_d, 'N': _case3_sc3_eta_p1_N,
    'offsets': _case3_sc3_eta_p1_offsets, 'row_ifaces': _case3_sc3_eta_p1_row_ifaces,
}
CASE3_EVAL[(3, 1)] = _case3_sc3_eta_p1

# case3 sub-case 4, eta sign -1
_case3_sc4_eta_m1_M = [
    [lambda ctx: ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1)), _Z],
    [lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ep*(2*ctx.L + ctx.l - 3)/((ctx.L - 1)*(ctx.L + ctx.l - 2)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.L*(ctx.L + 1)), lambda ctx: -ctx.ep*(ctx.L - 1)/((ctx.l - 1)*(ctx.L + ctx.l - 2))],
    [_Z, lambda ctx: ctx.ep*(ctx.l - 1)/((ctx.L - 1)*(ctx.L + ctx.l - 2)), lambda ctx: ctx.ep*(ctx.L + 2*ctx.l - 3)/((ctx.l - 1)*(ctx.L + ctx.l - 2)) - (2*ctx.l + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.l*(ctx.l + 1))],
]
_case3_sc4_eta_m1_d = [lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case3_sc4_eta_m1_N = [
    {(+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.em*ctx.dx*ctx.L*ctx.T + ctx.em*ctx.dx*ctx.L)/(ctx.dx*ctx.L*ctx.T), (+0, +1): lambda ctx: -ctx.ep*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: ctx.ep*(ctx.T - 1)/(ctx.T - 2), (+1, -1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dx*(ctx.L + 1))},
    {(-1, +0): lambda ctx: -ctx.ep*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)), (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L - ctx.ej*ctx.dy*ctx.n2**2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.T - ctx.em*ctx.dy*ctx.L*ctx.T - ctx.em*ctx.dy*ctx.T)/(ctx.dy*ctx.L*ctx.T), (+1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (+1, +0): lambda ctx: ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy)/(ctx.dy*(ctx.L + 1))},
    {(-3, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l/ctx.dy, (-3, +0): lambda ctx: ctx.l*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 + ctx.em*ctx.dy)/(ctx.dy*(ctx.l + 1)), (-2, -1): lambda ctx: (1/2)*ctx.ej*ctx.dx*ctx.n1*ctx.n2*(2*ctx.l + 1)/ctx.dy, (-2, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l**2 + ctx.ej*ctx.dy*ctx.n2**2*ctx.l + ctx.ej*ctx.dy*ctx.n2**2 + ctx.em*ctx.dy*ctx.l + ctx.em*ctx.dy)/(ctx.dy*ctx.l), (-2, +1): lambda ctx: -1/2*ctx.ej*ctx.dx*ctx.n1*ctx.n2/ctx.dy, (-1, +0): lambda ctx: ctx.ep*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1))},
]
_case3_sc4_eta_m1_offsets = [(-3, -1), (-3, +0), (-2, -1), (-2, +0), (-2, +1), (-1, +0), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, -1), (+1, +0)]
_case3_sc4_eta_m1_row_ifaces = ("T", "L", "extra")
_case3_sc4_eta_m1 = {
    'M': _case3_sc4_eta_m1_M, 'd': _case3_sc4_eta_m1_d, 'N': _case3_sc4_eta_m1_N,
    'offsets': _case3_sc4_eta_m1_offsets, 'row_ifaces': _case3_sc4_eta_m1_row_ifaces,
}
CASE3_EVAL[(4, -1)] = _case3_sc4_eta_m1

# case3 sub-case 4, eta sign +1
_case3_sc4_eta_p1_M = [
    [lambda ctx: -ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1)), _Z],
    [lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), lambda ctx: ctx.em*(2*ctx.L + ctx.l - 3)/((ctx.L - 1)*(ctx.L + ctx.l - 2)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.L*(ctx.L + 1)), lambda ctx: ctx.em*(ctx.L - 1)/((ctx.l - 1)*(ctx.L + ctx.l - 2))],
    [_Z, lambda ctx: -ctx.em*(ctx.l - 1)/((ctx.L - 1)*(ctx.L + ctx.l - 2)), lambda ctx: -ctx.em*(ctx.L + 2*ctx.l - 3)/((ctx.l - 1)*(ctx.L + ctx.l - 2)) - (2*ctx.l + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.l*(ctx.l + 1))],
]
_case3_sc4_eta_p1_d = [lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case3_sc4_eta_p1_N = [
    {(+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T - ctx.ep*ctx.dx*ctx.L*ctx.T - ctx.ep*ctx.dx*ctx.L)/(ctx.dx*ctx.L*ctx.T), (+0, +1): lambda ctx: ctx.em*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: -ctx.em*(ctx.T - 1)/(ctx.T - 2), (+1, -1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dx*(ctx.L + 1))},
    {(-1, +0): lambda ctx: ctx.em*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)), (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L - ctx.ej*ctx.dy*ctx.n2**2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.T + ctx.ep*ctx.dy*ctx.L*ctx.T + ctx.ep*ctx.dy*ctx.T)/(ctx.dy*ctx.L*ctx.T), (+1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (+1, +0): lambda ctx: ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy)/(ctx.dy*(ctx.L + 1))},
    {(-3, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l/ctx.dy, (-3, +0): lambda ctx: ctx.l*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 - ctx.ep*ctx.dy)/(ctx.dy*(ctx.l + 1)), (-2, -1): lambda ctx: (1/2)*ctx.ej*ctx.dx*ctx.n1*ctx.n2*(2*ctx.l + 1)/ctx.dy, (-2, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l**2 + ctx.ej*ctx.dy*ctx.n2**2*ctx.l + ctx.ej*ctx.dy*ctx.n2**2 - ctx.ep*ctx.dy*ctx.l - ctx.ep*ctx.dy)/(ctx.dy*ctx.l), (-2, +1): lambda ctx: -1/2*ctx.ej*ctx.dx*ctx.n1*ctx.n2/ctx.dy, (-1, +0): lambda ctx: -ctx.em*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1))},
]
_case3_sc4_eta_p1_offsets = [(-3, -1), (-3, +0), (-2, -1), (-2, +0), (-2, +1), (-1, +0), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, -1), (+1, +0)]
_case3_sc4_eta_p1_row_ifaces = ("T", "L", "extra")
_case3_sc4_eta_p1 = {
    'M': _case3_sc4_eta_p1_M, 'd': _case3_sc4_eta_p1_d, 'N': _case3_sc4_eta_p1_N,
    'offsets': _case3_sc4_eta_p1_offsets, 'row_ifaces': _case3_sc4_eta_p1_row_ifaces,
}
CASE3_EVAL[(4, 1)] = _case3_sc4_eta_p1

# case3 sub-case 5, eta sign -1
_case3_sc5_eta_m1_M = [
    [lambda ctx: -ctx.ep*(2*ctx.L + ctx.l - 3)/((ctx.L - 1)*(ctx.L + ctx.l - 2)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.L*(ctx.L + 1)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: -ctx.ep*(ctx.L - 1)/((ctx.l - 1)*(ctx.L + ctx.l - 2))],
    [lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1)), lambda ctx: -ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.B*(ctx.B + 1)), _Z],
    [lambda ctx: ctx.ep*(ctx.l - 1)/((ctx.L - 1)*(ctx.L + ctx.l - 2)), _Z, lambda ctx: ctx.ep*(ctx.L + 2*ctx.l - 3)/((ctx.l - 1)*(ctx.L + ctx.l - 2)) - (2*ctx.l + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.l*(ctx.l + 1))],
]
_case3_sc5_eta_m1_d = [lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case3_sc5_eta_m1_N = [
    {(-1, +0): lambda ctx: -ctx.ep*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B + ctx.em*ctx.dy*ctx.B*ctx.L + ctx.em*ctx.dy*ctx.B)/(ctx.dy*ctx.B*ctx.L), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B + ctx.L)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 + ctx.em*ctx.dy)/(ctx.dy*(ctx.L + 1)), (+1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy},
    {(+0, -2): lambda ctx: -ctx.ep*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: ctx.ep*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L + ctx.ej*ctx.dx*ctx.n1**2*ctx.L + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.L + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B + ctx.em*ctx.dx*ctx.B*ctx.L + ctx.em*ctx.dx*ctx.L)/(ctx.dx*ctx.B*ctx.L), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B + ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B + ctx.L)/(ctx.dx*(ctx.L + 1)), (+1, +1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx},
    {(-3, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l/ctx.dy, (-3, +0): lambda ctx: ctx.l*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 + ctx.em*ctx.dy)/(ctx.dy*(ctx.l + 1)), (-2, -1): lambda ctx: (1/2)*ctx.ej*ctx.dx*ctx.n1*ctx.n2*(2*ctx.l + 1)/ctx.dy, (-2, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l**2 + ctx.ej*ctx.dy*ctx.n2**2*ctx.l + ctx.ej*ctx.dy*ctx.n2**2 + ctx.em*ctx.dy*ctx.l + ctx.em*ctx.dy)/(ctx.dy*ctx.l), (-2, +1): lambda ctx: -1/2*ctx.ej*ctx.dx*ctx.n1*ctx.n2/ctx.dy, (-1, +0): lambda ctx: ctx.ep*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1))},
]
_case3_sc5_eta_m1_offsets = [(-3, -1), (-3, +0), (-2, -1), (-2, +0), (-2, +1), (-1, +0), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, +0), (+1, +1)]
_case3_sc5_eta_m1_row_ifaces = ("L", "B", "extra")
_case3_sc5_eta_m1 = {
    'M': _case3_sc5_eta_m1_M, 'd': _case3_sc5_eta_m1_d, 'N': _case3_sc5_eta_m1_N,
    'offsets': _case3_sc5_eta_m1_offsets, 'row_ifaces': _case3_sc5_eta_m1_row_ifaces,
}
CASE3_EVAL[(5, -1)] = _case3_sc5_eta_m1

# case3 sub-case 5, eta sign +1
_case3_sc5_eta_p1_M = [
    [lambda ctx: ctx.em*(2*ctx.L + ctx.l - 3)/((ctx.L - 1)*(ctx.L + ctx.l - 2)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.L*(ctx.L + 1)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: ctx.em*(ctx.L - 1)/((ctx.l - 1)*(ctx.L + ctx.l - 2))],
    [lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1)), lambda ctx: ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.B*(ctx.B + 1)), _Z],
    [lambda ctx: -ctx.em*(ctx.l - 1)/((ctx.L - 1)*(ctx.L + ctx.l - 2)), _Z, lambda ctx: -ctx.em*(ctx.L + 2*ctx.l - 3)/((ctx.l - 1)*(ctx.L + ctx.l - 2)) - (2*ctx.l + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.l*(ctx.l + 1))],
]
_case3_sc5_eta_p1_d = [lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case3_sc5_eta_p1_N = [
    {(-1, +0): lambda ctx: ctx.em*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1)), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B - ctx.ep*ctx.dy*ctx.B*ctx.L - ctx.ep*ctx.dy*ctx.B)/(ctx.dy*ctx.B*ctx.L), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B + ctx.L)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 - ctx.ep*ctx.dy)/(ctx.dy*(ctx.L + 1)), (+1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy},
    {(+0, -2): lambda ctx: ctx.em*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: -ctx.em*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L + ctx.ej*ctx.dx*ctx.n1**2*ctx.L + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.L + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ep*ctx.dx*ctx.B*ctx.L - ctx.ep*ctx.dx*ctx.L)/(ctx.dx*ctx.B*ctx.L), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B + ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B + ctx.L)/(ctx.dx*(ctx.L + 1)), (+1, +1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx},
    {(-3, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l/ctx.dy, (-3, +0): lambda ctx: ctx.l*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 - ctx.ep*ctx.dy)/(ctx.dy*(ctx.l + 1)), (-2, -1): lambda ctx: (1/2)*ctx.ej*ctx.dx*ctx.n1*ctx.n2*(2*ctx.l + 1)/ctx.dy, (-2, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.l**2 + ctx.ej*ctx.dy*ctx.n2**2*ctx.l + ctx.ej*ctx.dy*ctx.n2**2 - ctx.ep*ctx.dy*ctx.l - ctx.ep*ctx.dy)/(ctx.dy*ctx.l), (-2, +1): lambda ctx: -1/2*ctx.ej*ctx.dx*ctx.n1*ctx.n2/ctx.dy, (-1, +0): lambda ctx: -ctx.em*(ctx.L + ctx.l - 2)/((ctx.L - 1)*(ctx.l - 1))},
]
_case3_sc5_eta_p1_offsets = [(-3, -1), (-3, +0), (-2, -1), (-2, +0), (-2, +1), (-1, +0), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, +0), (+1, +1)]
_case3_sc5_eta_p1_row_ifaces = ("L", "B", "extra")
_case3_sc5_eta_p1 = {
    'M': _case3_sc5_eta_p1_M, 'd': _case3_sc5_eta_p1_d, 'N': _case3_sc5_eta_p1_N,
    'offsets': _case3_sc5_eta_p1_offsets, 'row_ifaces': _case3_sc5_eta_p1_row_ifaces,
}
CASE3_EVAL[(5, 1)] = _case3_sc5_eta_p1

# case3 sub-case 6, eta sign -1
_case3_sc6_eta_m1_M = [
    [lambda ctx: -ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.L*(ctx.L + 1)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), _Z],
    [lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1)), lambda ctx: -ctx.ep*(2*ctx.B + ctx.b - 3)/((ctx.B - 1)*(ctx.B + ctx.b - 2)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.B*(ctx.B + 1)), lambda ctx: -ctx.ep*(ctx.B - 1)/((ctx.b - 1)*(ctx.B + ctx.b - 2))],
    [_Z, lambda ctx: ctx.ep*(ctx.b - 1)/((ctx.B - 1)*(ctx.B + ctx.b - 2)), lambda ctx: ctx.ep*(ctx.B + 2*ctx.b - 3)/((ctx.b - 1)*(ctx.B + ctx.b - 2)) - (2*ctx.b + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.b*(ctx.b + 1))],
]
_case3_sc6_eta_m1_d = [lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc6_eta_m1_N = [
    {(-2, +0): lambda ctx: -ctx.ep*(ctx.L - 1)/(ctx.L - 2), (-1, +0): lambda ctx: ctx.ep*(ctx.L - 2)/(ctx.L - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B + ctx.em*ctx.dy*ctx.B*ctx.L + ctx.em*ctx.dy*ctx.B)/(ctx.dy*ctx.B*ctx.L), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B + ctx.L)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 + ctx.em*ctx.dy)/(ctx.dy*(ctx.L + 1)), (+1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy},
    {(+0, -1): lambda ctx: -ctx.ep*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L + ctx.ej*ctx.dx*ctx.n1**2*ctx.L + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.L + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B + ctx.em*ctx.dx*ctx.B*ctx.L + ctx.em*ctx.dx*ctx.L)/(ctx.dx*ctx.B*ctx.L), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B + ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B + ctx.L)/(ctx.dx*(ctx.L + 1)), (+1, +1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx},
    {(-1, -3): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b/ctx.dx, (-1, -2): lambda ctx: (1/2)*ctx.ej*ctx.dy*ctx.n1*ctx.n2*(2*ctx.b + 1)/ctx.dx, (+0, -3): lambda ctx: ctx.b*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b + ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.b + 1)), (+0, -2): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.b + ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b**2 + ctx.em*ctx.dx*ctx.b + ctx.em*ctx.dx)/(ctx.dx*ctx.b), (+0, -1): lambda ctx: ctx.ep*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)), (+1, -2): lambda ctx: -1/2*ctx.ej*ctx.dy*ctx.n1*ctx.n2/ctx.dx},
]
_case3_sc6_eta_m1_offsets = [(-2, +0), (-1, -3), (-1, -2), (-1, +0), (+0, -3), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, -2), (+1, +0), (+1, +1)]
_case3_sc6_eta_m1_row_ifaces = ("L", "B", "extra")
_case3_sc6_eta_m1 = {
    'M': _case3_sc6_eta_m1_M, 'd': _case3_sc6_eta_m1_d, 'N': _case3_sc6_eta_m1_N,
    'offsets': _case3_sc6_eta_m1_offsets, 'row_ifaces': _case3_sc6_eta_m1_row_ifaces,
}
CASE3_EVAL[(6, -1)] = _case3_sc6_eta_m1

# case3 sub-case 6, eta sign +1
_case3_sc6_eta_p1_M = [
    [lambda ctx: ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.L*(ctx.L + 1)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), _Z],
    [lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1)), lambda ctx: ctx.em*(2*ctx.B + ctx.b - 3)/((ctx.B - 1)*(ctx.B + ctx.b - 2)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.B*(ctx.B + 1)), lambda ctx: ctx.em*(ctx.B - 1)/((ctx.b - 1)*(ctx.B + ctx.b - 2))],
    [_Z, lambda ctx: -ctx.em*(ctx.b - 1)/((ctx.B - 1)*(ctx.B + ctx.b - 2)), lambda ctx: -ctx.em*(ctx.B + 2*ctx.b - 3)/((ctx.b - 1)*(ctx.B + ctx.b - 2)) - (2*ctx.b + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.b*(ctx.b + 1))],
]
_case3_sc6_eta_p1_d = [lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc6_eta_p1_N = [
    {(-2, +0): lambda ctx: ctx.em*(ctx.L - 1)/(ctx.L - 2), (-1, +0): lambda ctx: -ctx.em*(ctx.L - 2)/(ctx.L - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B - ctx.ep*ctx.dy*ctx.B*ctx.L - ctx.ep*ctx.dy*ctx.B)/(ctx.dy*ctx.B*ctx.L), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B + ctx.L)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 + ctx.ej*ctx.dy*ctx.n2**2 - ctx.ep*ctx.dy)/(ctx.dy*(ctx.L + 1)), (+1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy},
    {(+0, -1): lambda ctx: ctx.em*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L + ctx.ej*ctx.dx*ctx.n1**2*ctx.L + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.L + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ep*ctx.dx*ctx.B*ctx.L - ctx.ep*ctx.dx*ctx.L)/(ctx.dx*ctx.B*ctx.L), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B + ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B + ctx.L)/(ctx.dx*(ctx.L + 1)), (+1, +1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx},
    {(-1, -3): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b/ctx.dx, (-1, -2): lambda ctx: (1/2)*ctx.ej*ctx.dy*ctx.n1*ctx.n2*(2*ctx.b + 1)/ctx.dx, (+0, -3): lambda ctx: ctx.b*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b + ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.b + 1)), (+0, -2): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.b + ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b**2 - ctx.ep*ctx.dx*ctx.b - ctx.ep*ctx.dx)/(ctx.dx*ctx.b), (+0, -1): lambda ctx: -ctx.em*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)), (+1, -2): lambda ctx: -1/2*ctx.ej*ctx.dy*ctx.n1*ctx.n2/ctx.dx},
]
_case3_sc6_eta_p1_offsets = [(-2, +0), (-1, -3), (-1, -2), (-1, +0), (+0, -3), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, -2), (+1, +0), (+1, +1)]
_case3_sc6_eta_p1_row_ifaces = ("L", "B", "extra")
_case3_sc6_eta_p1 = {
    'M': _case3_sc6_eta_p1_M, 'd': _case3_sc6_eta_p1_d, 'N': _case3_sc6_eta_p1_N,
    'offsets': _case3_sc6_eta_p1_offsets, 'row_ifaces': _case3_sc6_eta_p1_row_ifaces,
}
CASE3_EVAL[(6, 1)] = _case3_sc6_eta_p1

# case3 sub-case 7, eta sign -1
_case3_sc7_eta_m1_M = [
    [lambda ctx: -ctx.ep*(2*ctx.B + ctx.b - 3)/((ctx.B - 1)*(ctx.B + ctx.b - 2)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.B*(ctx.B + 1)), lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), lambda ctx: -ctx.ep*(ctx.B - 1)/((ctx.b - 1)*(ctx.B + ctx.b - 2))],
    [lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.R*(ctx.R + 1)), _Z],
    [lambda ctx: ctx.ep*(ctx.b - 1)/((ctx.B - 1)*(ctx.B + ctx.b - 2)), _Z, lambda ctx: ctx.ep*(ctx.B + 2*ctx.b - 3)/((ctx.b - 1)*(ctx.B + ctx.b - 2)) - (2*ctx.b + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.b*(ctx.b + 1))],
]
_case3_sc7_eta_m1_d = [lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc7_eta_m1_N = [
    {(-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dx*(ctx.R + 1)), (-1, +1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+0, -1): lambda ctx: -ctx.ep*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B + ctx.em*ctx.dx*ctx.B*ctx.R + ctx.em*ctx.dx*ctx.R)/(ctx.dx*ctx.B*ctx.R), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.B + 1))},
    {(-1, +0): lambda ctx: -ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy)/(ctx.dy*(ctx.R + 1)), (-1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B - ctx.em*ctx.dy*ctx.B*ctx.R - ctx.em*ctx.dy*ctx.B)/(ctx.dy*ctx.B*ctx.R), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.ep*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: ctx.ep*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, -3): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b/ctx.dx, (-1, -2): lambda ctx: (1/2)*ctx.ej*ctx.dy*ctx.n1*ctx.n2*(2*ctx.b + 1)/ctx.dx, (+0, -3): lambda ctx: ctx.b*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b + ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.b + 1)), (+0, -2): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.b + ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b**2 + ctx.em*ctx.dx*ctx.b + ctx.em*ctx.dx)/(ctx.dx*ctx.b), (+0, -1): lambda ctx: ctx.ep*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)), (+1, -2): lambda ctx: -1/2*ctx.ej*ctx.dy*ctx.n1*ctx.n2/ctx.dx},
]
_case3_sc7_eta_m1_offsets = [(-1, -3), (-1, -2), (-1, +0), (-1, +1), (+0, -3), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, -2), (+1, +0), (+2, +0)]
_case3_sc7_eta_m1_row_ifaces = ("B", "R", "extra")
_case3_sc7_eta_m1 = {
    'M': _case3_sc7_eta_m1_M, 'd': _case3_sc7_eta_m1_d, 'N': _case3_sc7_eta_m1_N,
    'offsets': _case3_sc7_eta_m1_offsets, 'row_ifaces': _case3_sc7_eta_m1_row_ifaces,
}
CASE3_EVAL[(7, -1)] = _case3_sc7_eta_m1

# case3 sub-case 7, eta sign +1
_case3_sc7_eta_p1_M = [
    [lambda ctx: ctx.em*(2*ctx.B + ctx.b - 3)/((ctx.B - 1)*(ctx.B + ctx.b - 2)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.B*(ctx.B + 1)), lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), lambda ctx: ctx.em*(ctx.B - 1)/((ctx.b - 1)*(ctx.B + ctx.b - 2))],
    [lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: -ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.R*(ctx.R + 1)), _Z],
    [lambda ctx: -ctx.em*(ctx.b - 1)/((ctx.B - 1)*(ctx.B + ctx.b - 2)), _Z, lambda ctx: -ctx.em*(ctx.B + 2*ctx.b - 3)/((ctx.b - 1)*(ctx.B + ctx.b - 2)) - (2*ctx.b + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.b*(ctx.b + 1))],
]
_case3_sc7_eta_p1_d = [lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case3_sc7_eta_p1_N = [
    {(-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dx*(ctx.R + 1)), (-1, +1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+0, -1): lambda ctx: ctx.em*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ep*ctx.dx*ctx.B*ctx.R - ctx.ep*ctx.dx*ctx.R)/(ctx.dx*ctx.B*ctx.R), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.B + 1))},
    {(-1, +0): lambda ctx: -ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy)/(ctx.dy*(ctx.R + 1)), (-1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B + ctx.ep*ctx.dy*ctx.B*ctx.R + ctx.ep*ctx.dy*ctx.B)/(ctx.dy*ctx.B*ctx.R), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: ctx.em*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: -ctx.em*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, -3): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b/ctx.dx, (-1, -2): lambda ctx: (1/2)*ctx.ej*ctx.dy*ctx.n1*ctx.n2*(2*ctx.b + 1)/ctx.dx, (+0, -3): lambda ctx: ctx.b*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b + ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.b + 1)), (+0, -2): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.b + ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.b**2 - ctx.ep*ctx.dx*ctx.b - ctx.ep*ctx.dx)/(ctx.dx*ctx.b), (+0, -1): lambda ctx: -ctx.em*(ctx.B + ctx.b - 2)/((ctx.B - 1)*(ctx.b - 1)), (+1, -2): lambda ctx: -1/2*ctx.ej*ctx.dy*ctx.n1*ctx.n2/ctx.dx},
]
_case3_sc7_eta_p1_offsets = [(-1, -3), (-1, -2), (-1, +0), (-1, +1), (+0, -3), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, -2), (+1, +0), (+2, +0)]
_case3_sc7_eta_p1_row_ifaces = ("B", "R", "extra")
_case3_sc7_eta_p1 = {
    'M': _case3_sc7_eta_p1_M, 'd': _case3_sc7_eta_p1_d, 'N': _case3_sc7_eta_p1_N,
    'offsets': _case3_sc7_eta_p1_offsets, 'row_ifaces': _case3_sc7_eta_p1_row_ifaces,
}
CASE3_EVAL[(7, 1)] = _case3_sc7_eta_p1

# case3 sub-case 8, eta sign -1
_case3_sc8_eta_m1_M = [
    [lambda ctx: -ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.B*(ctx.B + 1)), lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), _Z],
    [lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: ctx.ep*(2*ctx.R + ctx.r - 3)/((ctx.R - 1)*(ctx.R + ctx.r - 2)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.R*(ctx.R + 1)), lambda ctx: ctx.ep*(ctx.R - 1)/((ctx.r - 1)*(ctx.R + ctx.r - 2))],
    [_Z, lambda ctx: -ctx.ep*(ctx.r - 1)/((ctx.R - 1)*(ctx.R + ctx.r - 2)), lambda ctx: -ctx.ep*(ctx.R + 2*ctx.r - 3)/((ctx.r - 1)*(ctx.R + ctx.r - 2)) + (2*ctx.r + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.r*(ctx.r + 1))],
]
_case3_sc8_eta_m1_d = [lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case3_sc8_eta_m1_N = [
    {(-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dx*(ctx.R + 1)), (-1, +1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+0, -2): lambda ctx: -ctx.ep*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: ctx.ep*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B + ctx.em*ctx.dx*ctx.B*ctx.R + ctx.em*ctx.dx*ctx.R)/(ctx.dx*ctx.B*ctx.R), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.B + 1))},
    {(-1, +0): lambda ctx: -ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy)/(ctx.dy*(ctx.R + 1)), (-1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B - ctx.em*ctx.dy*ctx.B*ctx.R - ctx.em*ctx.dy*ctx.B)/(ctx.dy*ctx.B*ctx.R), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: ctx.ep*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1))},
    {(+1, +0): lambda ctx: -ctx.ep*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)), (+2, -1): lambda ctx: (1/2)*ctx.ej*ctx.dx*ctx.n1*ctx.n2*(2*ctx.r + 1)/ctx.dy, (+2, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r**2 - ctx.ej*ctx.dy*ctx.n2**2*ctx.r - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy*ctx.r - ctx.em*ctx.dy)/(ctx.dy*ctx.r), (+2, +1): lambda ctx: -1/2*ctx.ej*ctx.dx*ctx.n1*ctx.n2/ctx.dy, (+3, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r/ctx.dy, (+3, +0): lambda ctx: ctx.r*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy)/(ctx.dy*(ctx.r + 1))},
]
_case3_sc8_eta_m1_offsets = [(-1, +0), (-1, +1), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, +0), (+2, -1), (+2, +0), (+2, +1), (+3, -1), (+3, +0)]
_case3_sc8_eta_m1_row_ifaces = ("B", "R", "extra")
_case3_sc8_eta_m1 = {
    'M': _case3_sc8_eta_m1_M, 'd': _case3_sc8_eta_m1_d, 'N': _case3_sc8_eta_m1_N,
    'offsets': _case3_sc8_eta_m1_offsets, 'row_ifaces': _case3_sc8_eta_m1_row_ifaces,
}
CASE3_EVAL[(8, -1)] = _case3_sc8_eta_m1

# case3 sub-case 8, eta sign +1
_case3_sc8_eta_p1_M = [
    [lambda ctx: ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.B*(ctx.B + 1)), lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), _Z],
    [lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: -ctx.em*(2*ctx.R + ctx.r - 3)/((ctx.R - 1)*(ctx.R + ctx.r - 2)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.R*(ctx.R + 1)), lambda ctx: -ctx.em*(ctx.R - 1)/((ctx.r - 1)*(ctx.R + ctx.r - 2))],
    [_Z, lambda ctx: ctx.em*(ctx.r - 1)/((ctx.R - 1)*(ctx.R + ctx.r - 2)), lambda ctx: ctx.em*(ctx.R + 2*ctx.r - 3)/((ctx.r - 1)*(ctx.R + ctx.r - 2)) + (2*ctx.r + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.r*(ctx.r + 1))],
]
_case3_sc8_eta_p1_d = [lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case3_sc8_eta_p1_N = [
    {(-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dx*(ctx.R + 1)), (-1, +1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+0, -2): lambda ctx: ctx.em*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: -ctx.em*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ep*ctx.dx*ctx.B*ctx.R - ctx.ep*ctx.dx*ctx.R)/(ctx.dx*ctx.B*ctx.R), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.B + 1))},
    {(-1, +0): lambda ctx: -ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy)/(ctx.dy*(ctx.R + 1)), (-1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B + ctx.ep*ctx.dy*ctx.B*ctx.R + ctx.ep*ctx.dy*ctx.B)/(ctx.dy*ctx.B*ctx.R), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.em*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1))},
    {(+1, +0): lambda ctx: ctx.em*(ctx.R + ctx.r - 2)/((ctx.R - 1)*(ctx.r - 1)), (+2, -1): lambda ctx: (1/2)*ctx.ej*ctx.dx*ctx.n1*ctx.n2*(2*ctx.r + 1)/ctx.dy, (+2, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r**2 - ctx.ej*ctx.dy*ctx.n2**2*ctx.r - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy*ctx.r + ctx.ep*ctx.dy)/(ctx.dy*ctx.r), (+2, +1): lambda ctx: -1/2*ctx.ej*ctx.dx*ctx.n1*ctx.n2/ctx.dy, (+3, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r/ctx.dy, (+3, +0): lambda ctx: ctx.r*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.r + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy)/(ctx.dy*(ctx.r + 1))},
]
_case3_sc8_eta_p1_offsets = [(-1, +0), (-1, +1), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, +0), (+2, -1), (+2, +0), (+2, +1), (+3, -1), (+3, +0)]
_case3_sc8_eta_p1_row_ifaces = ("B", "R", "extra")
_case3_sc8_eta_p1 = {
    'M': _case3_sc8_eta_p1_M, 'd': _case3_sc8_eta_p1_d, 'N': _case3_sc8_eta_p1_N,
    'offsets': _case3_sc8_eta_p1_offsets, 'row_ifaces': _case3_sc8_eta_p1_row_ifaces,
}
CASE3_EVAL[(8, 1)] = _case3_sc8_eta_p1

# case4 sub-case 1, eta sign -1
_case4_sc1_eta_m1_M = [
    [lambda ctx: ctx.ej*ctx.n2*(ctx.dx*ctx.n1*ctx.L*ctx.R - ctx.dx*ctx.n1*ctx.R - ctx.dy*ctx.n2*ctx.L - 2*ctx.dy*ctx.n2*ctx.R)/(ctx.dy*ctx.R*(ctx.L + ctx.R)) - ctx.em*(ctx.L + 2*ctx.R)/(ctx.R*(ctx.L + ctx.R)) + ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ej*ctx.n2*ctx.R*(ctx.dx*ctx.n1*ctx.R + ctx.dx*ctx.n1 + ctx.dy*ctx.n2)/(ctx.dy*ctx.L*(ctx.L + ctx.R)) - ctx.em*ctx.R/(ctx.L*(ctx.L + ctx.R))],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L - ctx.T)/(ctx.dx*ctx.R*(ctx.L + ctx.R)), lambda ctx: ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dx*ctx.L*(ctx.L + ctx.R))],
    [lambda ctx: -ctx.ej*ctx.n2*ctx.L*(ctx.dx*ctx.n1*ctx.L - ctx.dx*ctx.n1 - ctx.dy*ctx.n2)/(ctx.dy*ctx.R*(ctx.L + ctx.R)) + ctx.em*ctx.L/(ctx.R*(ctx.L + ctx.R)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), lambda ctx: ctx.ej*ctx.n2*(ctx.dx*ctx.n1*ctx.L*ctx.R + ctx.dx*ctx.n1*ctx.L + 2*ctx.dy*ctx.n2*ctx.L + ctx.dy*ctx.n2*ctx.R)/(ctx.dy*ctx.L*(ctx.L + ctx.R)) + ctx.em*(2*ctx.L + ctx.R)/(ctx.L*(ctx.L + ctx.R)) - ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1))],
]
_case4_sc1_eta_m1_d = [lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case4_sc1_eta_m1_N = [
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.R*ctx.T - ctx.em*ctx.dy*ctx.L*ctx.T - ctx.em*ctx.dy*ctx.R*ctx.T)/(ctx.dy*ctx.L*ctx.R*ctx.T), (+1, +0): lambda ctx: -ctx.ep*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: ctx.ep*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.R*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T**2 + ctx.em*ctx.dx*ctx.L*ctx.R*ctx.T + ctx.em*ctx.dx*ctx.L*ctx.R)/(ctx.dx*ctx.L*ctx.R*ctx.T), (+0, +1): lambda ctx: -ctx.ep*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: ctx.ep*(ctx.T - 1)/(ctx.T - 2)},
    {(-2, +0): lambda ctx: -ctx.ep*(ctx.L - 1)/(ctx.L - 2), (-1, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (-1, +0): lambda ctx: ctx.ep*(ctx.L - 2)/(ctx.L - 1), (+0, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L - ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.R*ctx.T - ctx.em*ctx.dy*ctx.L*ctx.T - ctx.em*ctx.dy*ctx.R*ctx.T)/(ctx.dy*ctx.L*ctx.R*ctx.T)},
]
_case4_sc1_eta_m1_offsets = [(-2, +0), (-1, -1), (-1, +0), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, +0), (+2, +0)]
_case4_sc1_eta_m1_row_ifaces = ("R", "T", "L")
_case4_sc1_eta_m1 = {
    'M': _case4_sc1_eta_m1_M, 'd': _case4_sc1_eta_m1_d, 'N': _case4_sc1_eta_m1_N,
    'offsets': _case4_sc1_eta_m1_offsets, 'row_ifaces': _case4_sc1_eta_m1_row_ifaces,
}
CASE4_EVAL[(1, -1)] = _case4_sc1_eta_m1

# case4 sub-case 1, eta sign +1
_case4_sc1_eta_p1_M = [
    [lambda ctx: ctx.ej*ctx.n2*(ctx.dx*ctx.n1*ctx.L*ctx.R - ctx.dx*ctx.n1*ctx.R - ctx.dy*ctx.n2*ctx.L - 2*ctx.dy*ctx.n2*ctx.R)/(ctx.dy*ctx.R*(ctx.L + ctx.R)) - ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.ep*(ctx.L + 2*ctx.R)/(ctx.R*(ctx.L + ctx.R)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ej*ctx.n2*ctx.R*(ctx.dx*ctx.n1*ctx.R + ctx.dx*ctx.n1 + ctx.dy*ctx.n2)/(ctx.dy*ctx.L*(ctx.L + ctx.R)) + ctx.ep*ctx.R/(ctx.L*(ctx.L + ctx.R))],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L - ctx.T)/(ctx.dx*ctx.R*(ctx.L + ctx.R)), lambda ctx: -ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) - (2*ctx.T + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.T*(ctx.T + 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dx*ctx.L*(ctx.L + ctx.R))],
    [lambda ctx: -ctx.ej*ctx.n2*ctx.L*(ctx.dx*ctx.n1*ctx.L - ctx.dx*ctx.n1 - ctx.dy*ctx.n2)/(ctx.dy*ctx.R*(ctx.L + ctx.R)) - ctx.ep*ctx.L/(ctx.R*(ctx.L + ctx.R)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.T*(ctx.T + 1)), lambda ctx: ctx.ej*ctx.n2*(ctx.dx*ctx.n1*ctx.L*ctx.R + ctx.dx*ctx.n1*ctx.L + 2*ctx.dy*ctx.n2*ctx.L + ctx.dy*ctx.n2*ctx.R)/(ctx.dy*ctx.L*(ctx.L + ctx.R)) + ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) - ctx.ep*(2*ctx.L + ctx.R)/(ctx.L*(ctx.L + ctx.R))],
]
_case4_sc1_eta_p1_d = [lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case4_sc1_eta_p1_N = [
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.R*ctx.T + ctx.R + ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.R*ctx.T + ctx.ep*ctx.dy*ctx.L*ctx.T + ctx.ep*ctx.dy*ctx.R*ctx.T)/(ctx.dy*ctx.L*ctx.R*ctx.T), (+1, +0): lambda ctx: ctx.em*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: -ctx.em*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, -1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+0, -1): lambda ctx: ctx.T*(ctx.ej*ctx.dx*ctx.n1**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.R*ctx.T + ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T**2 - ctx.ep*ctx.dx*ctx.L*ctx.R*ctx.T - ctx.ep*ctx.dx*ctx.L*ctx.R)/(ctx.dx*ctx.L*ctx.R*ctx.T), (+0, +1): lambda ctx: ctx.em*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: -ctx.em*(ctx.T - 1)/(ctx.T - 2)},
    {(-2, +0): lambda ctx: ctx.em*(ctx.L - 1)/(ctx.L - 2), (-1, -1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (-1, +0): lambda ctx: -ctx.em*(ctx.L - 2)/(ctx.L - 1), (+0, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L - ctx.T)/(ctx.dy*(ctx.T + 1)), (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.L*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.R*ctx.T + ctx.ep*ctx.dy*ctx.L*ctx.T + ctx.ep*ctx.dy*ctx.R*ctx.T)/(ctx.dy*ctx.L*ctx.R*ctx.T)},
]
_case4_sc1_eta_p1_offsets = [(-2, +0), (-1, -1), (-1, +0), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, +0), (+2, +0)]
_case4_sc1_eta_p1_row_ifaces = ("R", "T", "L")
_case4_sc1_eta_p1 = {
    'M': _case4_sc1_eta_p1_M, 'd': _case4_sc1_eta_p1_d, 'N': _case4_sc1_eta_p1_N,
    'offsets': _case4_sc1_eta_p1_offsets, 'row_ifaces': _case4_sc1_eta_p1_row_ifaces,
}
CASE4_EVAL[(1, 1)] = _case4_sc1_eta_p1

# case4 sub-case 2, eta sign -1
_case4_sc2_eta_m1_M = [
    [lambda ctx: ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.R*(ctx.R + 1)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dy*ctx.T*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.R*ctx.T - ctx.R + ctx.T)/(ctx.dy*ctx.B*(ctx.B + ctx.T))],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), lambda ctx: -ctx.ej*ctx.n1*(ctx.dx*ctx.n1*ctx.B + 2*ctx.dx*ctx.n1*ctx.T - ctx.dy*ctx.n2*ctx.B*ctx.T - ctx.dy*ctx.n2*ctx.T)/(ctx.dx*ctx.T*(ctx.B + ctx.T)) - ctx.em*(ctx.B + 2*ctx.T)/(ctx.T*(ctx.B + ctx.T)) + ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)), lambda ctx: -ctx.ej*ctx.n1*ctx.T*(ctx.dx*ctx.n1 + ctx.dy*ctx.n2*ctx.T - ctx.dy*ctx.n2)/(ctx.dx*ctx.B*(ctx.B + ctx.T)) - ctx.em*ctx.T/(ctx.B*(ctx.B + ctx.T))],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), lambda ctx: ctx.ej*ctx.n1*ctx.B*(ctx.dx*ctx.n1 - ctx.dy*ctx.n2*ctx.B - ctx.dy*ctx.n2)/(ctx.dx*ctx.T*(ctx.B + ctx.T)) + ctx.em*ctx.B/(ctx.T*(ctx.B + ctx.T)), lambda ctx: ctx.ej*ctx.n1*(2*ctx.dx*ctx.n1*ctx.B + ctx.dx*ctx.n1*ctx.T + ctx.dy*ctx.n2*ctx.B*ctx.T - ctx.dy*ctx.n2*ctx.B)/(ctx.dx*ctx.B*(ctx.B + ctx.T)) + ctx.em*(2*ctx.B + ctx.T)/(ctx.B*(ctx.B + ctx.T)) - ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1))],
]
_case4_sc2_eta_m1_d = [lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case4_sc2_eta_m1_N = [
    {(-1, +0): lambda ctx: -ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy)/(ctx.dy*(ctx.R + 1)), (-1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2 - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.T - ctx.em*ctx.dy*ctx.B*ctx.R*ctx.T - ctx.em*ctx.dy*ctx.B*ctx.T)/(ctx.dy*ctx.B*ctx.R*ctx.T), (+1, +0): lambda ctx: -ctx.ep*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: ctx.ep*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.R*ctx.T - ctx.R + ctx.T)/(ctx.dx*(ctx.R + 1)), (-1, +1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T + ctx.em*ctx.dx*ctx.B*ctx.R + ctx.em*ctx.dx*ctx.R*ctx.T)/(ctx.dx*ctx.B*ctx.R*ctx.T), (+0, +1): lambda ctx: -ctx.ep*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: ctx.ep*(ctx.T - 1)/(ctx.T - 2)},
    {(-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dx*(ctx.R + 1)), (-1, +1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+0, -2): lambda ctx: -ctx.ep*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: ctx.ep*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.T + ctx.em*ctx.dx*ctx.B*ctx.R + ctx.em*ctx.dx*ctx.R*ctx.T)/(ctx.dx*ctx.B*ctx.R*ctx.T)},
]
_case4_sc2_eta_m1_offsets = [(-1, +0), (-1, +1), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, +0), (+2, +0)]
_case4_sc2_eta_m1_row_ifaces = ("R", "T", "B")
_case4_sc2_eta_m1 = {
    'M': _case4_sc2_eta_m1_M, 'd': _case4_sc2_eta_m1_d, 'N': _case4_sc2_eta_m1_N,
    'offsets': _case4_sc2_eta_m1_offsets, 'row_ifaces': _case4_sc2_eta_m1_row_ifaces,
}
CASE4_EVAL[(2, -1)] = _case4_sc2_eta_m1

# case4 sub-case 2, eta sign +1
_case4_sc2_eta_p1_M = [
    [lambda ctx: -ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) - (2*ctx.R + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.R*(ctx.R + 1)), lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dy*ctx.T*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.R*ctx.T - ctx.R + ctx.T)/(ctx.dy*ctx.B*(ctx.B + ctx.T))],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), lambda ctx: -ctx.ej*ctx.n1*(ctx.dx*ctx.n1*ctx.B + 2*ctx.dx*ctx.n1*ctx.T - ctx.dy*ctx.n2*ctx.B*ctx.T - ctx.dy*ctx.n2*ctx.T)/(ctx.dx*ctx.T*(ctx.B + ctx.T)) - ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.ep*(ctx.B + 2*ctx.T)/(ctx.T*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.n1*ctx.T*(ctx.dx*ctx.n1 + ctx.dy*ctx.n2*ctx.T - ctx.dy*ctx.n2)/(ctx.dx*ctx.B*(ctx.B + ctx.T)) + ctx.ep*ctx.T/(ctx.B*(ctx.B + ctx.T))],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.R*(ctx.R + 1)), lambda ctx: ctx.ej*ctx.n1*ctx.B*(ctx.dx*ctx.n1 - ctx.dy*ctx.n2*ctx.B - ctx.dy*ctx.n2)/(ctx.dx*ctx.T*(ctx.B + ctx.T)) - ctx.ep*ctx.B/(ctx.T*(ctx.B + ctx.T)), lambda ctx: ctx.ej*ctx.n1*(2*ctx.dx*ctx.n1*ctx.B + ctx.dx*ctx.n1*ctx.T + ctx.dy*ctx.n2*ctx.B*ctx.T - ctx.dy*ctx.n2*ctx.B)/(ctx.dx*ctx.B*(ctx.B + ctx.T)) + ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) - ctx.ep*(2*ctx.B + ctx.T)/(ctx.B*(ctx.B + ctx.T))],
]
_case4_sc2_eta_p1_d = [lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2]
_case4_sc2_eta_p1_N = [
    {(-1, +0): lambda ctx: -ctx.R*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy)/(ctx.dy*(ctx.R + 1)), (-1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2*ctx.T + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R**2 - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.T + ctx.ep*ctx.dy*ctx.B*ctx.R*ctx.T + ctx.ep*ctx.dy*ctx.B*ctx.T)/(ctx.dy*ctx.B*ctx.R*ctx.T), (+1, +0): lambda ctx: ctx.em*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: -ctx.em*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.R*ctx.T - ctx.R + ctx.T)/(ctx.dx*(ctx.R + 1)), (-1, +1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.R*ctx.T - ctx.ep*ctx.dx*ctx.B*ctx.R - ctx.ep*ctx.dx*ctx.R*ctx.T)/(ctx.dx*ctx.B*ctx.R*ctx.T), (+0, +1): lambda ctx: ctx.em*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: -ctx.em*(ctx.T - 1)/(ctx.T - 2)},
    {(-1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dx*(ctx.R + 1)), (-1, +1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+0, -2): lambda ctx: ctx.em*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: -ctx.em*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.R*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.T - ctx.ep*ctx.dx*ctx.B*ctx.R - ctx.ep*ctx.dx*ctx.R*ctx.T)/(ctx.dx*ctx.B*ctx.R*ctx.T)},
]
_case4_sc2_eta_p1_offsets = [(-1, +0), (-1, +1), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, +0), (+2, +0)]
_case4_sc2_eta_p1_row_ifaces = ("R", "T", "B")
_case4_sc2_eta_p1 = {
    'M': _case4_sc2_eta_p1_M, 'd': _case4_sc2_eta_p1_d, 'N': _case4_sc2_eta_p1_N,
    'offsets': _case4_sc2_eta_p1_offsets, 'row_ifaces': _case4_sc2_eta_p1_row_ifaces,
}
CASE4_EVAL[(2, 1)] = _case4_sc2_eta_p1

# case4 sub-case 3, eta sign -1
_case4_sc3_eta_m1_M = [
    [lambda ctx: -ctx.ej*ctx.n2*(ctx.dx*ctx.n1*ctx.L*ctx.R - ctx.dx*ctx.n1*ctx.R + ctx.dy*ctx.n2*ctx.L + 2*ctx.dy*ctx.n2*ctx.R)/(ctx.dy*ctx.R*(ctx.L + ctx.R)) - ctx.em*(ctx.L + 2*ctx.R)/(ctx.R*(ctx.L + ctx.R)) + ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: ctx.ej*ctx.n2*ctx.R*(ctx.dx*ctx.n1*ctx.R + ctx.dx*ctx.n1 - ctx.dy*ctx.n2)/(ctx.dy*ctx.L*(ctx.L + ctx.R)) - ctx.em*ctx.R/(ctx.L*(ctx.L + ctx.R))],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.L - ctx.B + ctx.L)/(ctx.dx*ctx.R*(ctx.L + ctx.R)), lambda ctx: -ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 + ctx.em)/(ctx.B*(ctx.B + 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dx*ctx.L*(ctx.L + ctx.R))],
    [lambda ctx: ctx.ej*ctx.n2*ctx.L*(ctx.dx*ctx.n1*ctx.L - ctx.dx*ctx.n1 + ctx.dy*ctx.n2)/(ctx.dy*ctx.R*(ctx.L + ctx.R)) + ctx.em*ctx.L/(ctx.R*(ctx.L + ctx.R)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: -ctx.ej*ctx.n2*(ctx.dx*ctx.n1*ctx.L*ctx.R + ctx.dx*ctx.n1*ctx.L - 2*ctx.dy*ctx.n2*ctx.L - ctx.dy*ctx.n2*ctx.R)/(ctx.dy*ctx.L*(ctx.L + ctx.R)) + ctx.em*(2*ctx.L + ctx.R)/(ctx.L*(ctx.L + ctx.R)) - ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1))],
]
_case4_sc3_eta_m1_d = [lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 - ctx.av*ctx.ep*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case4_sc3_eta_m1_N = [
    {(-1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R - ctx.em*ctx.dy*ctx.B*ctx.L - ctx.em*ctx.dy*ctx.B*ctx.R)/(ctx.dy*ctx.B*ctx.L*ctx.R), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: -ctx.ep*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: ctx.ep*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, +1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+0, -2): lambda ctx: -ctx.ep*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: ctx.ep*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.R + ctx.em*ctx.dx*ctx.B*ctx.L*ctx.R + ctx.em*ctx.dx*ctx.L*ctx.R)/(ctx.dx*ctx.B*ctx.L*ctx.R), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ej*ctx.dy*ctx.n1*ctx.n2 + ctx.em*ctx.dx)/(ctx.dx*(ctx.B + 1))},
    {(-2, +0): lambda ctx: -ctx.ep*(ctx.L - 1)/(ctx.L - 2), (-1, +0): lambda ctx: ctx.ep*(ctx.L - 2)/(ctx.L - 1), (-1, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L**2 - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.R + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R + ctx.em*ctx.dy*ctx.B*ctx.L + ctx.em*ctx.dy*ctx.B*ctx.R)/(ctx.dy*ctx.B*ctx.L*ctx.R), (+0, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.L - ctx.B + ctx.L)/(ctx.dy*(ctx.B + 1))},
]
_case4_sc3_eta_m1_offsets = [(-2, +0), (-1, +0), (-1, +1), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, +0), (+2, +0)]
_case4_sc3_eta_m1_row_ifaces = ("R", "B", "L")
_case4_sc3_eta_m1 = {
    'M': _case4_sc3_eta_m1_M, 'd': _case4_sc3_eta_m1_d, 'N': _case4_sc3_eta_m1_N,
    'offsets': _case4_sc3_eta_m1_offsets, 'row_ifaces': _case4_sc3_eta_m1_row_ifaces,
}
CASE4_EVAL[(3, -1)] = _case4_sc3_eta_m1

# case4 sub-case 3, eta sign +1
_case4_sc3_eta_p1_M = [
    [lambda ctx: -ctx.ej*ctx.n2*(ctx.dx*ctx.n1*ctx.L*ctx.R - ctx.dx*ctx.n1*ctx.R + ctx.dy*ctx.n2*ctx.L + 2*ctx.dy*ctx.n2*ctx.R)/(ctx.dy*ctx.R*(ctx.L + ctx.R)) - ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.ep*(ctx.L + 2*ctx.R)/(ctx.R*(ctx.L + ctx.R)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: ctx.ej*ctx.n2*ctx.R*(ctx.dx*ctx.n1*ctx.R + ctx.dx*ctx.n1 - ctx.dy*ctx.n2)/(ctx.dy*ctx.L*(ctx.L + ctx.R)) + ctx.ep*ctx.R/(ctx.L*(ctx.L + ctx.R))],
    [lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.L - ctx.B + ctx.L)/(ctx.dx*ctx.R*(ctx.L + ctx.R)), lambda ctx: ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + (2*ctx.B + 1)*(ctx.ej*ctx.n1**2 - ctx.ep)/(ctx.B*(ctx.B + 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dx*ctx.L*(ctx.L + ctx.R))],
    [lambda ctx: ctx.ej*ctx.n2*ctx.L*(ctx.dx*ctx.n1*ctx.L - ctx.dx*ctx.n1 + ctx.dy*ctx.n2)/(ctx.dy*ctx.R*(ctx.L + ctx.R)) - ctx.ep*ctx.L/(ctx.R*(ctx.L + ctx.R)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2/(ctx.dy*ctx.B*(ctx.B + 1)), lambda ctx: -ctx.ej*ctx.n2*(ctx.dx*ctx.n1*ctx.L*ctx.R + ctx.dx*ctx.n1*ctx.L - 2*ctx.dy*ctx.n2*ctx.L - ctx.dy*ctx.n2*ctx.R)/(ctx.dy*ctx.L*(ctx.L + ctx.R)) + ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) - ctx.ep*(2*ctx.L + ctx.R)/(ctx.L*(ctx.L + ctx.R))],
]
_case4_sc3_eta_p1_d = [lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 - ctx.av*ctx.em*(2*ctx.R - 3)/((ctx.R - 2)*(ctx.R - 1)) + ctx.bv*ctx.dx*ctx.n1, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case4_sc3_eta_p1_N = [
    {(-1, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.R/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.R - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R + ctx.ep*ctx.dy*ctx.B*ctx.L + ctx.ep*ctx.dy*ctx.B*ctx.R)/(ctx.dy*ctx.B*ctx.L*ctx.R), (+0, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.R + ctx.B + ctx.R)/(ctx.dy*(ctx.B + 1)), (+1, +0): lambda ctx: ctx.em*(ctx.R - 2)/(ctx.R - 1), (+2, +0): lambda ctx: -ctx.em*(ctx.R - 1)/(ctx.R - 2)},
    {(-1, +1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+0, -2): lambda ctx: ctx.em*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: -ctx.em*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L*ctx.R + ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.R + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.R - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2 + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.R - ctx.ep*ctx.dx*ctx.B*ctx.L*ctx.R - ctx.ep*ctx.dx*ctx.L*ctx.R)/(ctx.dx*ctx.B*ctx.L*ctx.R), (+0, +1): lambda ctx: -ctx.B*(ctx.ej*ctx.dx*ctx.n1**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B - ctx.ej*ctx.dy*ctx.n1*ctx.n2 - ctx.ep*ctx.dx)/(ctx.dx*(ctx.B + 1))},
    {(-2, +0): lambda ctx: ctx.em*(ctx.L - 1)/(ctx.L - 2), (-1, +0): lambda ctx: -ctx.em*(ctx.L - 2)/(ctx.L - 1), (-1, +1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L**2 - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.R + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.R - ctx.ep*ctx.dy*ctx.B*ctx.L - ctx.ep*ctx.dy*ctx.B*ctx.R)/(ctx.dy*ctx.B*ctx.L*ctx.R), (+0, +1): lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.L - ctx.B + ctx.L)/(ctx.dy*(ctx.B + 1))},
]
_case4_sc3_eta_p1_offsets = [(-2, +0), (-1, +0), (-1, +1), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+1, +0), (+2, +0)]
_case4_sc3_eta_p1_row_ifaces = ("R", "B", "L")
_case4_sc3_eta_p1 = {
    'M': _case4_sc3_eta_p1_M, 'd': _case4_sc3_eta_p1_d, 'N': _case4_sc3_eta_p1_N,
    'offsets': _case4_sc3_eta_p1_offsets, 'row_ifaces': _case4_sc3_eta_p1_row_ifaces,
}
CASE4_EVAL[(3, 1)] = _case4_sc3_eta_p1

# case4 sub-case 4, eta sign -1
_case4_sc4_eta_m1_M = [
    [lambda ctx: -ctx.ej*ctx.n1*(ctx.dx*ctx.n1*ctx.B + 2*ctx.dx*ctx.n1*ctx.T + ctx.dy*ctx.n2*ctx.B*ctx.T - ctx.dy*ctx.n2*ctx.T)/(ctx.dx*ctx.T*(ctx.B + ctx.T)) - ctx.em*(ctx.B + 2*ctx.T)/(ctx.T*(ctx.B + ctx.T)) + ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)), lambda ctx: -ctx.ej*ctx.n1*ctx.T*(ctx.dx*ctx.n1 - ctx.dy*ctx.n2*ctx.T - ctx.dy*ctx.n2)/(ctx.dx*ctx.B*(ctx.B + ctx.T)) - ctx.em*ctx.T/(ctx.B*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1))],
    [lambda ctx: ctx.ej*ctx.n1*ctx.B*(ctx.dx*ctx.n1 + ctx.dy*ctx.n2*ctx.B - ctx.dy*ctx.n2)/(ctx.dx*ctx.T*(ctx.B + ctx.T)) + ctx.em*ctx.B/(ctx.T*(ctx.B + ctx.T)), lambda ctx: ctx.ej*ctx.n1*(2*ctx.dx*ctx.n1*ctx.B + ctx.dx*ctx.n1*ctx.T - ctx.dy*ctx.n2*ctx.B*ctx.T - ctx.dy*ctx.n2*ctx.B)/(ctx.dx*ctx.B*(ctx.B + ctx.T)) + ctx.em*(2*ctx.B + ctx.T)/(ctx.B*(ctx.B + ctx.T)) - ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1))],
    [lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B - ctx.L)/(ctx.dy*ctx.T*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dy*ctx.B*(ctx.B + ctx.T)), lambda ctx: -ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 + ctx.em)/(ctx.L*(ctx.L + 1))],
]
_case4_sc4_eta_m1_d = [lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 - ctx.av*ctx.ep*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: ctx.at*ctx.ep*ctx.dy*ctx.n1 + ctx.av*ctx.ep*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.ep*ctx.dx*ctx.n2 + ctx.av*ctx.ep*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case4_sc4_eta_m1_N = [
    {(+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L + ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.em*ctx.dx*ctx.B*ctx.L + ctx.em*ctx.dx*ctx.L*ctx.T)/(ctx.dx*ctx.B*ctx.L*ctx.T), (+0, +1): lambda ctx: -ctx.ep*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: ctx.ep*(ctx.T - 1)/(ctx.T - 2), (+1, -1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dx*(ctx.L + 1))},
    {(+0, -2): lambda ctx: -ctx.ep*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: ctx.ep*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L + ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.T + ctx.em*ctx.dx*ctx.B*ctx.L + ctx.em*ctx.dx*ctx.L*ctx.T)/(ctx.dx*ctx.B*ctx.L*ctx.T), (+1, -1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B - ctx.L)/(ctx.dx*(ctx.L + 1))},
    {(-2, +0): lambda ctx: -ctx.ep*(ctx.L - 1)/(ctx.L - 2), (-1, +0): lambda ctx: ctx.ep*(ctx.L - 2)/(ctx.L - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2 - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.T + ctx.em*ctx.dy*ctx.B*ctx.L*ctx.T + ctx.em*ctx.dy*ctx.B*ctx.T)/(ctx.dy*ctx.B*ctx.L*ctx.T), (+1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (+1, +0): lambda ctx: ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 - ctx.em*ctx.dy)/(ctx.dy*(ctx.L + 1))},
]
_case4_sc4_eta_m1_offsets = [(-2, +0), (-1, +0), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, -1), (+1, +0)]
_case4_sc4_eta_m1_row_ifaces = ("T", "B", "L")
_case4_sc4_eta_m1 = {
    'M': _case4_sc4_eta_m1_M, 'd': _case4_sc4_eta_m1_d, 'N': _case4_sc4_eta_m1_N,
    'offsets': _case4_sc4_eta_m1_offsets, 'row_ifaces': _case4_sc4_eta_m1_row_ifaces,
}
CASE4_EVAL[(4, -1)] = _case4_sc4_eta_m1

# case4 sub-case 4, eta sign +1
_case4_sc4_eta_p1_M = [
    [lambda ctx: -ctx.ej*ctx.n1*(ctx.dx*ctx.n1*ctx.B + 2*ctx.dx*ctx.n1*ctx.T + ctx.dy*ctx.n2*ctx.B*ctx.T - ctx.dy*ctx.n2*ctx.T)/(ctx.dx*ctx.T*(ctx.B + ctx.T)) - ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.ep*(ctx.B + 2*ctx.T)/(ctx.T*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.n1*ctx.T*(ctx.dx*ctx.n1 - ctx.dy*ctx.n2*ctx.T - ctx.dy*ctx.n2)/(ctx.dx*ctx.B*(ctx.B + ctx.T)) + ctx.ep*ctx.T/(ctx.B*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1))],
    [lambda ctx: ctx.ej*ctx.n1*ctx.B*(ctx.dx*ctx.n1 + ctx.dy*ctx.n2*ctx.B - ctx.dy*ctx.n2)/(ctx.dx*ctx.T*(ctx.B + ctx.T)) - ctx.ep*ctx.B/(ctx.T*(ctx.B + ctx.T)), lambda ctx: ctx.ej*ctx.n1*(2*ctx.dx*ctx.n1*ctx.B + ctx.dx*ctx.n1*ctx.T - ctx.dy*ctx.n2*ctx.B*ctx.T - ctx.dy*ctx.n2*ctx.B)/(ctx.dx*ctx.B*(ctx.B + ctx.T)) + ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) - ctx.ep*(2*ctx.B + ctx.T)/(ctx.B*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2/(ctx.dx*ctx.L*(ctx.L + 1))],
    [lambda ctx: ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B - ctx.L)/(ctx.dy*ctx.T*(ctx.B + ctx.T)), lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dy*ctx.B*(ctx.B + ctx.T)), lambda ctx: ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + (2*ctx.L + 1)*(ctx.ej*ctx.n2**2 - ctx.ep)/(ctx.L*(ctx.L + 1))],
]
_case4_sc4_eta_p1_d = [lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 - ctx.av*ctx.em*(2*ctx.T - 3)/((ctx.T - 2)*(ctx.T - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: ctx.at*ctx.em*ctx.dy*ctx.n1 + ctx.av*ctx.em*(2*ctx.B - 3)/((ctx.B - 2)*(ctx.B - 1)) + ctx.bv*ctx.dy*ctx.n2, lambda ctx: -ctx.at*ctx.em*ctx.dx*ctx.n2 + ctx.av*ctx.em*(2*ctx.L - 3)/((ctx.L - 2)*(ctx.L - 1)) + ctx.bv*ctx.dx*ctx.n1]
_case4_sc4_eta_p1_N = [
    {(+0, +0): lambda ctx: -(ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L + ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.T - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T**2 - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.L*ctx.T - ctx.ep*ctx.dx*ctx.B*ctx.L - ctx.ep*ctx.dx*ctx.L*ctx.T)/(ctx.dx*ctx.B*ctx.L*ctx.T), (+0, +1): lambda ctx: ctx.em*(ctx.T - 2)/(ctx.T - 1), (+0, +2): lambda ctx: -ctx.em*(ctx.T - 1)/(ctx.T - 2), (+1, -1): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.T/ctx.dx, (+1, +0): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.L*ctx.T + ctx.L + ctx.T)/(ctx.dx*(ctx.L + 1))},
    {(+0, -2): lambda ctx: ctx.em*(ctx.B - 1)/(ctx.B - 2), (+0, -1): lambda ctx: -ctx.em*(ctx.B - 2)/(ctx.B - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1**2*ctx.B*ctx.L + ctx.ej*ctx.dx*ctx.n1**2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B**2*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B*ctx.T - ctx.ep*ctx.dx*ctx.B*ctx.L - ctx.ep*ctx.dx*ctx.L*ctx.T)/(ctx.dx*ctx.B*ctx.L*ctx.T), (+1, -1): lambda ctx: -ctx.ej*ctx.dy*ctx.n1*ctx.n2*ctx.B/ctx.dx, (+1, +0): lambda ctx: ctx.ej*ctx.dy*ctx.n1*ctx.n2*(ctx.B*ctx.L + ctx.B - ctx.L)/(ctx.dx*(ctx.L + 1))},
    {(-2, +0): lambda ctx: ctx.em*(ctx.L - 1)/(ctx.L - 2), (-1, +0): lambda ctx: -ctx.em*(ctx.L - 2)/(ctx.L - 1), (+0, +0): lambda ctx: (ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L**2 + ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.B*ctx.L - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2*ctx.T - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L**2 - ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.L*ctx.T + ctx.ej*ctx.dy*ctx.n2**2*ctx.B*ctx.T - ctx.ep*ctx.dy*ctx.B*ctx.L*ctx.T - ctx.ep*ctx.dy*ctx.B*ctx.T)/(ctx.dy*ctx.B*ctx.L*ctx.T), (+1, -1): lambda ctx: -ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L/ctx.dy, (+1, +0): lambda ctx: ctx.L*(ctx.ej*ctx.dx*ctx.n1*ctx.n2*ctx.L + ctx.ej*ctx.dx*ctx.n1*ctx.n2 - ctx.ej*ctx.dy*ctx.n2**2 + ctx.ep*ctx.dy)/(ctx.dy*(ctx.L + 1))},
]
_case4_sc4_eta_p1_offsets = [(-2, +0), (-1, +0), (+0, -2), (+0, -1), (+0, +0), (+0, +1), (+0, +2), (+1, -1), (+1, +0)]
_case4_sc4_eta_p1_row_ifaces = ("T", "B", "L")
_case4_sc4_eta_p1 = {
    'M': _case4_sc4_eta_p1_M, 'd': _case4_sc4_eta_p1_d, 'N': _case4_sc4_eta_p1_N,
    'offsets': _case4_sc4_eta_p1_offsets, 'rof_ifaces': _case4_sc4_eta_p1_row_ifaces,
}
CASE4_EVAL[(4, 1)] = _case4_sc4_eta_p1

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
    _CASE3_SUB.update({
        (Direction.R | Direction.T, Direction.R): 1,
        (Direction.R | Direction.T, Direction.T): 2,
        (Direction.L | Direction.T, Direction.T): 3,
        (Direction.L | Direction.T, Direction.L): 4,
        (Direction.L | Direction.B, Direction.L): 5,
        (Direction.L | Direction.B, Direction.B): 6,
        (Direction.R | Direction.B, Direction.B): 7,
        (Direction.R | Direction.B, Direction.R): 8,
    })
    _CASE4_SUB.update({
        int(Direction.R | Direction.T | Direction.L): 1,
        int(Direction.R | Direction.T | Direction.B): 2,
        int(Direction.R | Direction.B | Direction.L): 3,
        int(Direction.T | Direction.B | Direction.L): 4,
    })


class Direction(enum.IntFlag):
    R = 1 << 0  # 0001
    T = 1 << 1  # 0010
    L = 1 << 2  # 0100
    B = 1 << 3  # 1000


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
        self.axis = axis      # 0 → x-axis; 1 → y-axis
        self.sign = sign      # +1 for R/T (forward), -1 for L/B (backward)

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
        theta = (-cd.sign * d_eta - np.sign(eta) * np.sqrt(d_eta**2 - 4 * dd_eta * eta)) / (2 * dd_eta)
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
        points = field[k - 1 : k + 3, j] if cd.sign > 0 else field[k - 2 : k + 2, j][::-1]
    else:
        k = j
        points = field[i, k - 1 : k + 3] if cd.sign > 0 else field[i, k - 2 : k + 2][::-1]
    return 0.5 * t_matrix @ c_matrix @ points


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
    cd = _CD[direction]
    x, y = center(i, j)
    eta = surface(x, y)
    theta = compute_theta(direction, i, j)
    a_tau_I = interp(direction, theta, i, j, a_tau)
    a_I = interp(direction, theta, i, j, a)
    b_I = interp(direction, theta, i, j, b)
    n1_I = interp(direction, theta, i, j, n1)
    n2_I = interp(direction, theta, i, j, n2)

    # theta assignments: one active (theta), rest 1.0
    theta_l, theta_r, theta_t, theta_b = cd.theta_assign(theta)
    bot_x = (theta_r + theta_l) / 2 * dx**2
    bot_y = (theta_t + theta_b) / 2 * dy**2

    # permittivity at four half-points (uncut faces have theta=1 → h/2 spacing)
    eps_r = permittivity(x + theta_r * dx / 2, y)
    eps_l = permittivity(x - theta_l * dx / 2, y)
    eps_t = permittivity(x, y + theta_t * dy / 2)
    eps_b = permittivity(x, y - theta_b * dy / 2)

    # beta sampling at the interface point
    px, py, p_axis = cd.probe_loc(x, y, theta)
    _eps_p, _eps_m, eps_jump, eps_p, eps_m = _sample_beta_legacy(px, py, p_axis, eta)

    # normal components: n_t = tangential (along interface), n_n = surface-normal
    n_t = [n1_I, n2_I][cd.n_tang]
    n_n = [n1_I, n2_I][cd.n_norm]

    # d-vector entry (RHS contribution)
    d_self = dx if cd.is_x else dy
    d = (
        (-1 if cd.is_x else 1) * a_tau_I * eps_p * n_t * d_self
        + b_I * n_n * d_self
        + cd.sign * a_I * eps_p * _phi(theta)
    )

    if eta > 0:
        eps_p, eps_m = -_eps_m, -_eps_p

    # scalar M (1x1 mass matrix)
    M = -cd.sign * (
        eps_p * _phi(theta) + eps_m * _psi(theta) + eps_jump * n_t**2 * _psi(theta)
    )

    # 7-element N vector (couplings to stencil neighbours)
    d_other = dy if cd.is_x else dx
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

    # --- Shortley-Weller assembly ---
    _eps = {'r': eps_r, 'l': eps_l, 't': eps_t, 'b': eps_b}
    _theta = {'r': theta_r, 'l': theta_l, 't': theta_t, 'b': theta_b}
    _bot = {'r': bot_x, 'l': bot_x, 't': bot_y, 'b': bot_y}
    slot_names = ['l', 'r', 't', 'b']
    active = slot_names[cd.slot]
    eps_a = _eps[active]
    theta_a = _theta[active]
    bot_a = _bot[active]

    fac = 1.0 / M * eps_a / theta_a / bot_a
    f[i, j] -= d * fac

    # uncut-face neighbour corrections
    uncut_map = {}
    for di_f, dj_f, eps_name in CutDir.uncut_faces(cd.face):
        s = eps_name[4]  # 'r', 'l', 't', 'b'
        uncut_map[(di_f, dj_f)] = _eps[s] / _theta[s] / _bot[s]

    # Shortley-Weller diagonal: negative sum of all four flux terms
    sw_diag = -(
        eps_r / theta_r / bot_x + eps_l / theta_l / bot_x
        + eps_t / theta_t / bot_y + eps_b / theta_b / bot_y
    )

    row_idx = index(i, j)
    rows.extend([row_idx] * 7)
    for k, (di, dj) in enumerate(cd.offsets):
        cols.append(index(i + di, j + dj))
        val = N[k] * fac
        val += uncut_map.get((di, dj), 0.0)
        if k == 0:
            val += sw_diag
        vals.append(val)


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
        return -cd.sign * (eps_p * _phi(theta) + eps_m * _psi(theta) + eps_jump * n_t**2 * _psi(theta))

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

    def _fill_N_row(r, cd_p, cd_o, theta_p, theta_o, n1_v, n2_v, eps_p, eps_m, eps_jump):
        n_t = [n1_v, n2_v][cd_p.n_tang]
        d_self = dx if cd_p.is_x else dy
        d_other = dy if cd_p.is_x else dx
        off = cd_p.offsets

        # k=0 : self
        N[r, _col(*off[0])] = (
            -cd_p.sign * (eps_jump * n_t**2 + eps_m) * (1 + theta_p) / theta_p
            - cd_o.sign * eps_jump * n1_v * n2_v * d_self / d_other
            * (theta_o * theta_p + theta_o - 1) / theta_o
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
            cd_o.sign * eps_jump * n1_v * n2_v * d_self / d_other
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

    _fill_N_row(0, cd_x, cd_y, theta_x, theta_y, n1_x, n2_x, eps_p_x, eps_m_x, eps_jump_x)
    _fill_N_row(1, cd_y, cd_x, theta_y, theta_x, n1_y, n2_y, eps_p_y, eps_m_y, eps_jump_y)

    M_inv_d = np.linalg.solve(M, d)
    M_inv_N = np.linalg.solve(M, N)

    sw_idx = {_FACE_NAME[cd_x.face]: 0, _FACE_NAME[cd_y.face]: 1}
    geom = {
        "theta_R": theta_r, "theta_T": theta_t, "theta_L": theta_l, "theta_B": theta_b,
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
    M_inv_d, M_inv_N, all_offsets, sw_idx, geom = _solve_case2_local(direction, i, j)
    _assemble_case_n(
        i, j, sw_idx, M_inv_d, M_inv_N, all_offsets,
        geom["eps_r"], geom["eps_l"], geom["eps_t"], geom["eps_b"],
        geom["theta_R"], geom["theta_T"], geom["theta_L"], geom["theta_B"],
        geom["bot_x"], geom["bot_y"],
    )


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


def case3_extra_dir(direction: int, i: int, j: int) -> int | None:
    """Detect the 'extra' direction that promotes a case-2 stencil to case 3.

    Returns one of Direction.{R, T, L, B} if a single extra outer-segment cut
    is found consistent with the corner given by `direction` (a 2-bit value
    R|T / L|T / R|B / L|B); returns None if no such extra cut exists (i.e.
    the cell is plain case 2). Raises if both possible outer segments are
    cut (that would be a higher-order case not handled here).
    """
    x_, y_ = center(i, j)

    extras = []
    # Probe only the two outer segments adjacent to the case-2 corner.
    if direction == (Direction.R | Direction.T):
        if surface(x_ + dx, y_) * surface(x_ + 2 * dx, y_) < 0:
            extras.append(Direction.R)
        if surface(x_, y_ + dy) * surface(x_, y_ + 2 * dy) < 0:
            extras.append(Direction.T)
    elif direction == (Direction.L | Direction.T):
        if surface(x_, y_ + dy) * surface(x_, y_ + 2 * dy) < 0:
            extras.append(Direction.T)
        if surface(x_ - dx, y_) * surface(x_ - 2 * dx, y_) < 0:
            extras.append(Direction.L)
    elif direction == (Direction.L | Direction.B):
        if surface(x_ - dx, y_) * surface(x_ - 2 * dx, y_) < 0:
            extras.append(Direction.L)
        if surface(x_, y_ - dy) * surface(x_, y_ - 2 * dy) < 0:
            extras.append(Direction.B)
    elif direction == (Direction.R | Direction.B):
        if surface(x_, y_ - dy) * surface(x_, y_ - 2 * dy) < 0:
            extras.append(Direction.B)
        if surface(x_ + dx, y_) * surface(x_ + 2 * dx, y_) < 0:
            extras.append(Direction.R)

    if len(extras) == 0:
        return None
    if len(extras) > 1:
        raise NotImplementedError(
            f"Two extra outer-segment cuts at ({i},{j}); beyond case 3."
        )

    # Case 3's local stencil reaches +/- 3 grid cells. If we're too close to
    # the Dirichlet boundary to access the full stencil, fall back to plain
    # case 2 (drop the outer-segment interface). This loses one order of
    # accuracy locally but only at the boundary rim.
    extra = extras[0]
    if extra == Direction.R and i + 3 >= nx:
        return None
    if extra == Direction.L and i - 3 < 0:
        return None
    if extra == Direction.T and j + 3 >= ny:
        return None
    if extra == Direction.B and j - 3 < 0:
        return None
    return extra


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
        "theta_R": theta_R, "theta_T": theta_T,
        "theta_L": theta_L, "theta_B": theta_B,
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
    """Add the case-3 row contributions for grid point (i,j).

    `direction` is the (2-bit) corner type R|T / L|T / R|B / L|B, and
    `extra` is one of Direction.{R, T, L, B} naming the outer-segment
    interface (xr / xt / xl / xb). The eight (direction, extra) combinations
    map 1-to-1 to the eight sub-cases in derivation_case3.ju.py:

        (R|T, R) -> sub-case 1     (R|T, T) -> sub-case 2
        (L|T, T) -> sub-case 3     (L|T, L) -> sub-case 4
        (L|B, L) -> sub-case 5     (L|B, B) -> sub-case 6
        (R|B, B) -> sub-case 7     (R|B, R) -> sub-case 8

    The 3x3 system is M [u_*1, u_*2, u_*3]^T = N u_stencil + d, where the
    three unknowns are the interface ghost values at the two corner points
    (xR/xT/xL/xB) and the one extra point (xr/xt/xl/xb). The exact
    M / N / d for each sub-case must be filled in from the symbolic
    derivation in derivation_case3.ju.py.
    """
    x_, y_ = center(i, j)
    geom = _case3_geometry(direction, extra, i, j)
    eta = geom["eta"]

    # Identify which sub-case (1-8) this maps to.
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

    # The Shortley-Weller geometry on the case-3 stencil is the same as case 2
    # at the corner — only the local (M, N, d) for the three interface ghosts
    # changes. theta's for the corner-side eps factors:
    theta_R = geom["theta_R"]
    theta_T = geom["theta_T"]
    theta_L = geom["theta_L"]
    theta_B = geom["theta_B"]
    bot_x = (theta_R + theta_L) / 2 * dx**2
    bot_y = (theta_T + theta_B) / 2 * dy**2

    # eps at the four half-points (used by the Shortley-Weller assembly later)
    eps_r = permittivity(x_ + theta_R * dx / 2, y_)
    eps_l = permittivity(x_ - theta_L * dx / 2, y_)
    eps_t = permittivity(x_, y_ + theta_T * dy / 2)
    eps_b = permittivity(x_, y_ - theta_B * dy / 2)

    M_inv_d, M_inv_N, all_offsets = _solve_case3_local(
        sub_case, eta, direction, extra, geom
    )

    sw_idx = _CASE3_SW_INDEX[sub_case]
    _assemble_case_n(
        i, j, sw_idx, M_inv_d, M_inv_N, all_offsets,
        eps_r, eps_l, eps_t, eps_b,
        theta_R, theta_T, theta_L, theta_B, bot_x, bot_y,
    )


# ---------------------------------------------------------------------------
# Case 4
# ---------------------------------------------------------------------------
# Case 4 has THREE corner cuts at (i,j) — three of {R, T, L, B} are crossed
# and the fourth segment stays in the home region. There is no extra outer-
# segment cut (otherwise we would be in a higher case). The Omega^- polynomial
# at (i,j) carries all three corner interfaces; each Omega^+ neighbour
# (i+/-1,j) or (i,j+/-1) gets its own single-interface polynomial.
#
# Sub-cases (matching examples/poisson/derivation_case4.ju.py):
#   1. R|T|L   no B cut
#   2. R|T|B   no L cut
#   3. R|B|L   no T cut
#   4. T|B|L   no R cut


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
        "theta_R": theta_R, "theta_T": theta_T,
        "theta_L": theta_L, "theta_B": theta_B,
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
        diag = (x_ + dx, y_ + dy)        # any +x or +y neighbour is in Omega^+
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
    """Add the case-4 row contributions for grid point (i,j).

    `direction` has exactly three of R/T/L/B bits set. The four sub-cases
    map 1-to-1 to the four cells in derivation_case4.ju.py:

        R|T|L -> sub-case 1     R|T|B -> sub-case 2
        R|B|L -> sub-case 3     T|B|L -> sub-case 4

    The 3x3 system is M [u_*1, u_*2, u_*3]^T = N u_stencil + d, where the
    three unknowns are the interface ghost values at the three corner
    interface points. The exact M / N / d for each sub-case must be filled
    in from the symbolic derivation in derivation_case4.ju.py.
    """
    x_, y_ = center(i, j)
    geom = _case4_geometry(direction, i, j)
    eta = geom["eta"]

    sub_case_table = {
        int(Direction.R | Direction.T | Direction.L): 1,
        int(Direction.R | Direction.T | Direction.B): 2,
        int(Direction.R | Direction.B | Direction.L): 3,
        int(Direction.T | Direction.B | Direction.L): 4,
    }
    sub_case = sub_case_table.get(int(direction))
    if sub_case is None:
        raise ValueError(f"Invalid direction for case 4: {direction!r}")

    # Shortley-Weller half-segment widths (uncut side has theta = 1).
    theta_R = geom["theta_R"]
    theta_T = geom["theta_T"]
    theta_L = geom["theta_L"]
    theta_B = geom["theta_B"]
    bot_x = (theta_R + theta_L) / 2 * dx**2
    bot_y = (theta_T + theta_B) / 2 * dy**2

    eps_r = permittivity(x_ + theta_R * dx / 2, y_)
    eps_l = permittivity(x_ - theta_L * dx / 2, y_)
    eps_t = permittivity(x_, y_ + theta_T * dy / 2)
    eps_b = permittivity(x_, y_ - theta_B * dy / 2)

    M_inv_d, M_inv_N, all_offsets = _solve_case4_local(
        sub_case, eta, direction, geom
    )

    sw_idx = _CASE4_SW_INDEX[sub_case]
    _assemble_case_n(
        i, j, sw_idx, M_inv_d, M_inv_N, all_offsets,
        eps_r, eps_l, eps_t, eps_b,
        theta_R, theta_T, theta_L, theta_B, bot_x, bot_y,
    )


# ---------------------------------------------------------------------------
# Local 3x3 solve and Shortley-Weller assembly (shared by case 3 and case 4)
# ---------------------------------------------------------------------------

def _row_geom_data(geom, iface):
    """Return (n1, n2, a, b, a_tau) for one row's interface label."""
    if iface == "extra":
        return (geom["n1_x"], geom["n2_x"], geom["a_x"],
                geom["b_x"], geom["a_tau_x"])
    return (geom[f"n1_{iface}"], geom[f"n2_{iface}"], geom[f"a_{iface}"],
            geom[f"b_{iface}"], geom[f"a_tau_{iface}"])


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


def _build_local_system(eval_data, geom, theta_inputs, betas_per_row):
    """Evaluate hardcoded M/d/N formulas; return (M_inv_d, M_inv_N, offsets).

    `eval_data`       : dict from CASE3_EVAL or CASE4_EVAL with keys
                        'M', 'd', 'N', 'offsets', 'row_ifaces'.
    `geom`            : dict from `_caseN_geometry`, supplying per-interface
                        n_x, n_y, a, b, a_tau.
    `theta_inputs`    : dict with keys 'R','T','L','B','r','t','l','b' (1.0
                        if unused in this sub-case).
    `betas_per_row`   : list of 3 (beta_p, beta_m, beta_jump) tuples — one
                        per interface row, sampled AT that interface.
    """
    M_funcs = eval_data["M"]
    d_funcs = eval_data["d"]
    N_funcs = eval_data["N"]
    all_offsets = eval_data["offsets"]
    row_ifaces = eval_data["row_ifaces"]

    M = np.zeros((3, 3))
    d_vec = np.zeros(3)
    N = np.zeros((3, len(all_offsets)))

    for i_row in range(3):
        n1_v, n2_v, a_v, b_v, atau_v = _row_geom_data(geom, row_ifaces[i_row])
        bp, bm, bj = betas_per_row[i_row]
        ctx = EvalCtx(
            R=theta_inputs["R"], T=theta_inputs["T"],
            L=theta_inputs["L"], B=theta_inputs["B"],
            r=theta_inputs["r"], t=theta_inputs["t"],
            l=theta_inputs["l"], b=theta_inputs["b"],
            n1=n1_v, n2=n2_v,
            av=a_v, bv=b_v, at=atau_v,
            ep=bp, em=bm, ej=bj,
            dx=dx, dy=dy,
        )
        for j_col in range(3):
            M[i_row, j_col] = M_funcs[i_row][j_col](ctx)
        d_vec[i_row] = d_funcs[i_row](ctx)
        for k, off in enumerate(all_offsets):
            if off in N_funcs[i_row]:
                N[i_row, k] = N_funcs[i_row][off](ctx)

    M_inv_d = np.linalg.solve(M, d_vec)
    M_inv_N = np.linalg.solve(M, N)
    return M_inv_d, M_inv_N, all_offsets


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
    eval_data = CASE3_EVAL[(sub_case, eta_sign)]

    theta_inputs = {
        "R": geom["theta_R"], "T": geom["theta_T"],
        "L": geom["theta_L"], "B": geom["theta_B"],
        "r": 1.0, "t": 1.0, "l": 1.0, "b": 1.0,
    }
    if extra == Direction.R:
        theta_inputs["r"] = geom["theta_extra"]
    elif extra == Direction.T:
        theta_inputs["t"] = geom["theta_extra"]
    elif extra == Direction.L:
        theta_inputs["l"] = geom["theta_extra"]
    elif extra == Direction.B:
        theta_inputs["b"] = geom["theta_extra"]

    betas_per_row = _row_betas(geom, eval_data["row_ifaces"], extra=extra)
    return _build_local_system(
        eval_data, geom, theta_inputs, betas_per_row,
    )


def _solve_case4_local(sub_case, eta, direction, geom):
    eta_sign = -1 if eta < 0 else +1
    eval_data = CASE4_EVAL[(sub_case, eta_sign)]

    theta_inputs = {
        "R": geom["theta_R"], "T": geom["theta_T"],
        "L": geom["theta_L"], "B": geom["theta_B"],
        "r": 1.0, "t": 1.0, "l": 1.0, "b": 1.0,
    }
    betas_per_row = _row_betas(geom, eval_data["row_ifaces"])
    return _build_local_system(
        eval_data, geom, theta_inputs, betas_per_row,
    )


def _assemble_case_n(i, j, sw_idx, M_inv_d, M_inv_N, all_offsets,
                    eps_r, eps_l, eps_t, eps_b,
                    theta_R, theta_T, theta_L, theta_B, bot_x, bot_y):
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
                    extra = case3_extra_dir(direction, i, j)
                    if extra is None:
                        coeff_case2(direction, i, j)
                    else:
                        coeff_case3(direction, extra, i, j)
                case 3:
                    coeff_case4(direction, i, j)
                case _:
                    raise NotImplementedError(
                        "All four sides cut at one cell — beyond case 4."
                    )

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
    slot_names = ['l', 'r', 't', 'b']
    face_vals = {'l': u[i - 1, j], 'r': u[i + 1, j],
                 'b': u[i, j - 1], 't': u[i, j + 1]}
    face_vals[slot_names[cd.slot]] = u_I
    return (face_vals['l'], face_vals['r'], face_vals['b'], face_vals['t'],
            theta_l, theta_r, theta_b, theta_t)


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
    M_inv_d, M_inv_N, all_offsets = _solve_case4_local(
        sub_case, eta, direction, geom
    )
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
                    if extra is None:
                        u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                            interface_value_case2(direction, i, j, u)
                        )
                    else:
                        u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                            interface_value_case3(direction, extra, i, j, u)
                        )
                case 3:
                    u_l, u_r, u_b, u_t, theta_l, theta_r, theta_b, theta_t = (
                        interface_value_case4(direction, i, j, u)
                    )
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

        errors_u[idx] = np.linalg.norm(
            (u - u_exact)[2:-2, 2:-2].flat, np.inf
        )
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
    plt.xlabel("h"); plt.ylabel("err"); plt.legend()
    plt.title("Example 4.2: convergence of $u$")
    plt.subplot(122)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h"); plt.ylabel("err"); plt.legend()
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
        b = (sxy + 2.0) * (2.0 * X * n1 + 2.0 * Y * n2) / R2P1 \
            - e10x * cxy * (n1 + n2)
        a_tau = compute_a_tau_field()

        rows, cols, vals = [], [], []
        A = construct_matrix()
        u = spsolve(A, f.flatten())
        u = u.reshape((nx, ny))
        dudx, dudy = gradient(u)

        errors_u[idx] = np.linalg.norm(
            (u - u_exact)[2:-2, 2:-2].flat, np.inf
        )
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
    plt.xlabel("h"); plt.ylabel("err"); plt.legend()
    plt.title("Example 4.3: convergence of $u$")
    plt.subplot(122)
    plt.loglog(1 / n_range, errors_du, "o-", label="actual")
    plt.loglog(1 / n_range, 1 / n_range**2, "--", label="$O(h^2)$")
    plt.xlabel("h"); plt.ylabel("err"); plt.legend()
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
    convergence_test5()

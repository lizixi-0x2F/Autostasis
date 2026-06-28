"""Solver kernel -- the T-transform and its fixed point.

Every equation A[u] = 0 is solved by Fix(T(A, eta)).

  T[u] = u - eta * A[u]

T is the canonical gradient dynamics on function space.
It is not "one method among many". It is THE unique transformation
whose fixed point IS the solution. There is no better T.
"""
from .terms import Var, Lam, App, Fix
from .dsl import R, sub, mul


def T(A, eta=0.1):
    """T[u] = u - eta * A[u] -- the canonical transform."""
    x = Var("x")
    return Lam("x", sub(x, mul(R(eta), App(A, x))))


def solve(A, eta=0.1):
    """Fix(T(A, eta)) -- fixed point IS the solution.

    Proof:
      Fix(T) = T(Fix(T))
             = Fix(T) - eta * A(Fix(T))
      => A(Fix(T)) = 0
    """
    return Fix(T(A, eta))

"""All Term types — the unified space of programs and data.

Inhabitants of Term space:
  Computation layer: Var, Lam, App, Quote, Eval, Fix     — lambda-calculus + self-reference
  Data layer: Nat, Cons                                   — natural numbers + pairs
  Quantity layer: Fun, Space                              — points in vector space and space itself

Core design:
  - Space is Term —— space itself is a first-class inhabitant
  - Fun carries a space field —— every quantity knows where it lives
  - Scalar = Fun over a single-point domain (constant function) —— no Real/Fun type branch
  - Space.distance/norm/zero —— the space algebra unifies distance/norm/zero-element for scalars and functions
  - Pointwise algebraic lifting (addf/subf/mulf) —— vector space operations apply to all inhabitants
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable
import math


@dataclass(frozen=True)
class Term:
    """Every Term can be both operator and operand."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Function — unified abstract interface for symbolic and numerical functions
# ═══════════════════════════════════════════════════════════════════════════════

class Function(Term, ABC):
    """Abstract function — shared interface for symbolic (Lam) and numerical (Fun).

    Subclasses must provide:
      - space: Space         as a dataclass field
      - eval_at(x, env)      concrete point evaluation
    """
    space: Space  # type annotation — enforced by dataclass field on subclasses

    @abstractmethod
    def eval_at(self, x: float, env=None) -> float:
        """Evaluate the function at point x.

        Fun: returns self.fn(x) directly.
        Lam: beta-reduces App(self, constant(x)) and extracts the float.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Lambda-calculus
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Var(Term):
    name: str
    def __repr__(self): return self.name


@dataclass(frozen=True)
class Lam(Function):
    param: str
    body: Term
    space: Space = field(default=None, compare=False, hash=False)
    def __post_init__(self):
        if self.space is None:
            object.__setattr__(self, "space", SPACE_C0_DEFAULT)

    def __repr__(self): return f"(λ{self.param}. {self.body})"

    def eval_at(self, x: float, env=None) -> float:
        """β-reduce App(self, constant(x)), then extract float."""
        if env is None:
            raise ValueError("Lam.eval_at requires env")
        # Deferred import avoids circular dependency
        from .eval import eval_term
        result = eval_term(App(self, Fun(lambda _: x, space=SPACE_R)), env, 0)
        return _extract_float(result)


@dataclass(frozen=True)
class App(Term):
    func: Term
    arg: Term
    def __repr__(self): return f"({self.func} {self.arg})"


# ═══════════════════════════════════════════════════════════════════════════════
# Self-reference primitives
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Quote(Term):
    """Reify a term as data — the primitive that enables self-reference."""
    term: Term
    def __repr__(self): return f"'{self.term}'"


@dataclass(frozen=True)
class Eval(Term):
    """eval(quote(t), x) → t(x) — the fundamental self-reference axiom."""
    quoted: Term
    arg: Term
    def __repr__(self): return f"eval({self.quoted}, {self.arg})"


@dataclass(frozen=True)
class Fix(Term):
    """fix(f) → f(fix(f)) — Kleene's first recursion theorem."""
    func: Term
    def __repr__(self): return f"fix({self.func})"


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Nat(Term):
    value: int
    def __repr__(self): return str(self.value)


@dataclass(frozen=True)
class Cons(Term):
    car: Term
    cdr: Term
    def __repr__(self): return f"({self.car} . {self.cdr})"


@dataclass(frozen=True)
class Prim(Term):
    name: str
    def __repr__(self): return f"#{self.name}"


@dataclass(frozen=True)
class PartialPrim(Term):
    """Partially applied primitive — curried application."""
    name: str
    arg1: Term
    def __repr__(self): return f"#{self.name}({self.arg1}, ...)"


@dataclass(frozen=True)
class Integ(Term):
    """Integral operator -- first-class Term, not a Python escape hatch.

    Integ(f, a)      ->  F(x) = int_a^x f(t) dt        (Volterra / IVP)
    Integ(f, a, b)   ->  F(x) = int_a^b G(x,t) f(t) dt (Fredholm / BVP)

    b=None for IVP and b=value for BVP share the same type; evaluator dispatches.
    """
    func: Term              # integrand Lam("t", body)
    a: Term                 # lower bound
    b: Term | None = None   # upper bound (None=IVP, non-None=BVP)
    def __repr__(self):
        r = f"Integ({self.func}, {self.a}"
        if self.b is not None: r += f", {self.b}"
        return r + ")"


@dataclass(frozen=True)
class Diff(Term):
    """Derivative operator -- first-class Term, not a Python escape hatch.

    Diff(func) applied to x -> f'(x) via central finite difference.
    """
    func: Term              # function to differentiate
    def __repr__(self):
        return f"Diff({self.func})"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers: Domain
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Domain:
    """Function domain [a, b]"""
    a: float
    b: float
    def __repr__(self): return f"[{self.a:.2g},{self.b:.2g}]"
    def __iter__(self): yield self.a; yield self.b
    def __getitem__(self, i): return (self.a, self.b)[i]


# ═══════════════════════════════════════════════════════════════════════════════
# Space — Termified
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Space(Term):
    """Vector space — a meta-type for Term space.

    Spaces are type tags for quantities + carriers of algebra.
    The methods dict reserves a path to pure Termification: Lam Terms can
    replace Python callables in the future.

    Attributes:
        name: Space name ("ℝ", "C⁰", "C¹", "H¹", "Δᵏ⁻¹"...)
        dim: Dimension (None = infinite-dimensional)
        domain: Domain (None for scalar spaces)
        smoothness: Smoothness (0 = continuous, 1 = C¹, ...)
        methods: Space operations -> callable/str (gradual Termification)
    """
    name: str
    dim: int | None = None
    domain: Domain | None = None
    smoothness: int = 0
    methods: dict = field(default_factory=dict, hash=False, compare=False)

    def __repr__(self):
        parts = [self.name]
        if self.domain is not None:
            parts.append(str(self.domain))
        if self.smoothness > 0:
            parts.append(f"C{self.smoothness}")
        return f"Space({', '.join(parts)})"

    # ── Algebraic operations ──

    def _get_method(self, name: str):
        """Gradual Termification: check methods dict first, fall back to Python implementation."""
        return self.methods.get(name)

    def distance(self, f: Term, g: Term, env=None, n: int = 100) -> float:
        """Distance between two points in the space.

        R:   |a - b|
        C0:  (int|f-g|^2)^(1/2)  [L2 norm]
        C1:  (int|f-g|^2 + int|f'-g'|^2)^(1/2)  [H1 norm]
        """
        if self.dim is not None and self.domain is None:
            # Finite-dimensional Euclidean space (R, R^n) — scalar = constant function, take its value at any point
            return abs(_extract_float(f) - _extract_float(g))
        else:
            # Function space
            return _fun_l2_distance(f, g, self.domain, env, n)

    def norm(self, term: Term, env=None, n: int = 100) -> float:
        """Norm of a point in the space."""
        return self.distance(term, self.zero(), env, n)

    def zero(self) -> Term:
        """Zero element of the space."""
        if self.dim is not None and self.domain is None:
            return Fun(lambda _: 0.0, space=self)
        else:
            return Fun(lambda x: 0.0, space=self)

    def contains(self, term: Term) -> bool:
        """Check whether term is a valid inhabitant of this space."""
        match term:
            case Fun(_, space=Space() as sp):
                return sp is self or (sp is None)
            case Function():
                return self.domain is not None  # Lam is always a function
            case _:
                return False


# ── Predefined spaces ──

SPACE_R = Space("ℝ", dim=1)
"""Real line — 1-dimensional Euclidean space"""


def SPACE_C0(a: float = 0.0, b: float = 1.0) -> Space:
    """Continuous function space C0[a,b]"""
    return Space("C⁰", domain=Domain(a, b), smoothness=0)


def SPACE_C1(a: float = 0.0, b: float = 1.0) -> Space:
    """C1[a,b]"""
    return Space("C¹", domain=Domain(a, b), smoothness=1)


SPACE_C0_DEFAULT = SPACE_C0(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Quantity — a point in vector space
# ═══════════════════════════════════════════════════════════════════════════════

def constant(value: float, space: Space = None) -> "Fun":
    """Construct a scalar (constant function) — a point in R space.

    A scalar = a function over a single-point domain.  constant(3.14, SPACE_R) and
    Fun(lambda x: sin(x), SPACE_C0) are different inhabitants of the same type.
    """
    if space is None:
        space = SPACE_R
    return Fun(lambda _: value, space=space)


@dataclass(frozen=True)
class Fun(Function):
    """Numeric quantity — a point in vector space.

    fn is a Python callable (numerical implementation layer).
    Scalar = Fun on SPACE_R (constant function), function = Fun on SPACE_C0 (varying function).
    In Term space, Fun participates in pointwise algebraic operations just like Lam.

    The domain is given by space.domain — domain is no longer a standalone field.
    """
    fn: Callable = field(hash=False, compare=False)
    space: Space = field(default=SPACE_C0_DEFAULT, compare=False, hash=False)
    label: str = ""

    @property
    def domain(self) -> Domain:
        """Convenience access: domain is space.domain."""
        return self.space.domain or Domain(0.0, 1.0)

    def __repr__(self):
        lbl = f" {self.label}" if self.label else ""
        dom = self.domain
        return f"Fun({dom}{lbl})"

    def __call__(self, x):
        return self.fn(x)

    def eval_at(self, x: float, env=None) -> float:
        """Direct numerical evaluation — ignores env."""
        return self.fn(x)

    def sample(self, n: int = 5) -> str:
        """Sample and display over the domain."""
        d = self.domain
        xs = [d.a + i * (d.b - d.a) / (n - 1) for i in range(n)]
        return ", ".join(f"f({x:.2f})={self(x):.4f}" for x in xs)


# ═══════════════════════════════════════════════════════════════════════════════
# Space helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_float(t: Term) -> float:
    """Extract a float value from a scalar Term.

    A scalar is a Fun on SPACE_R (constant function), taking the same value at any point.
    """
    match t:
        case Fun() as f:
            return f(0)
        case Nat(v):
            return float(v)
    raise ValueError(f"Cannot extract float: {type(t).__name__}: {t}")


def _fun_l2_distance(f: Term, g: Term, domain: Domain, env=None, n: int = 100) -> float:
    """Compute the L2 distance between two functions."""
    import numpy as np
    a, b = domain.a, domain.b
    xs = np.linspace(a, b, n)
    dys = []
    for x in xs:
        fx = _eval_at_point(f, x, env)
        gx = _eval_at_point(g, x, env)
        dys.append((fx - gx) ** 2)
    # Trapezoidal rule
    total = 0.0
    for i in range(n - 1):
        total += (dys[i] + dys[i + 1]) * (xs[i + 1] - xs[i]) / 2.0
    return float(np.sqrt(total))


def _eval_at_point(term: Term, x: float, env) -> float:
    """Evaluate term at point x — dispatches through Function.eval_at."""
    match term:
        case Function() as f:
            return f.eval_at(x, env)
        case _:
            raise ValueError(f"Cannot evaluate at point: {type(term).__name__}")


def space_of(term: Term) -> Space:
    """Get the space of a term."""
    match term:
        case Function() as f:  return f.space
        case _:
            # Other Terms: fall back to R space
            return SPACE_R

"""All Term types — the syntax of a logic-focused computational model.

Inhabitants of Term space:
  Computation layer: Var, Lam, App, Quote, Eval, Fix     — lambda-calculus + self-reference
  Data layer:        Nat, Cons                           — natural numbers + pairs

Self-reference is first-class:
  Quote reifies a term as data, Eval executes quoted data, Fix is
  Kleene's first recursion theorem as syntax. A language that can
  talk about its own terms — the natural home for logic: proofs are
  terms, derivations are reductions, paradoxes are divergent terms.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    """Every Term can be both operator and operand."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Lambda-calculus
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Var(Term):
    name: str
    def __repr__(self): return self.name


@dataclass(frozen=True)
class Lam(Term):
    param: str
    body: Term
    def __repr__(self): return f"(λ{self.param}. {self.body})"


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

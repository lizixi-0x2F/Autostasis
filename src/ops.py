"""Syntactic operations: free variables, capture-avoiding substitution."""
from __future__ import annotations
import itertools
from .terms import (Term, Var, Lam, App, Quote, Eval, Fix, Nat, Fun,
                     Prim, PartialPrim, Cons, Integ, Diff)


def free_vars(term: Term) -> set[str]:
    match term:
        case Var(name):              return {name}
        case Lam(param, body):       return free_vars(body) - {param}
        case App(func, arg):         return free_vars(func) | free_vars(arg)
        case Quote(t):               return free_vars(t)
        case Eval(q, a):             return free_vars(q) | free_vars(a)
        case Fix(f):                 return free_vars(f)
        case Cons(car, cdr):         return free_vars(car) | free_vars(cdr)
        case PartialPrim(_, arg1):   return free_vars(arg1)
        case Integ(func, a, b):
            fv = free_vars(func) | free_vars(a)
            return fv | free_vars(b) if b is not None else fv
        case Diff(func):             return free_vars(func)
        case Fun(_, _, _) | Nat(_) | Prim(_):
            return set()
        case _:                      return set()


def _fresh_var(avoid: set[str], prefix: str = "x") -> str:
    for i in itertools.count():
        name = f"{prefix}{i}" if i > 0 else prefix
        if name not in avoid:
            return name
    raise RuntimeError("unreachable")


def substitute(term: Term, var: str, replacement: Term) -> Term:
    """Capture-avoiding substitution: term[var := replacement]."""
    match term:
        case Var(name):
            return replacement if name == var else term

        case Lam(param, body):
            if param == var:
                return term
            if param in free_vars(replacement):
                fv = free_vars(body) | free_vars(replacement)
                new_param = _fresh_var(fv, param)
                new_body = substitute(body, param, Var(new_param))
                return Lam(new_param, substitute(new_body, var, replacement))
            return Lam(param, substitute(body, var, replacement))

        case App(func, arg):
            return App(substitute(func, var, replacement),
                       substitute(arg, var, replacement))

        case Quote(t):
            return Quote(substitute(t, var, replacement))

        case Eval(q, a):
            return Eval(substitute(q, var, replacement),
                        substitute(a, var, replacement))

        case Fix(f):
            return Fix(substitute(f, var, replacement))

        case Cons(car, cdr):
            return Cons(substitute(car, var, replacement),
                        substitute(cdr, var, replacement))

        case PartialPrim(name, arg1):
            return PartialPrim(name, substitute(arg1, var, replacement))

        case Integ(func, a, b):
            new_func = substitute(func, var, replacement)
            new_a = substitute(a, var, replacement)
            new_b = substitute(b, var, replacement) if b is not None else None
            return Integ(new_func, new_a, new_b)

        case Diff(func):
            return Diff(substitute(func, var, replacement))

        case Fun(_, _, _) | Nat(_) | Prim(_):
            return term

        case _:
            return term

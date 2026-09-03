"""Core evaluator — evaluates terms to weak head normal form.

A logic-focused evaluator:

  - Nat arithmetic: add, sub (monus), mult, iszero, eq, eq_nat
  - Cons destructuring: car, cdr
  - Term introspection: is_var/is_lam/... , get_body/get_func/...
  - Term construction: mk_lam, mk_app
  - Self-reference: fix(f) → f(fix(f)); eval(quote(t), x) → t x
  - Church booleans and Y/Z combinators in the default environment

The evaluator is lazy (WHNF): a term with no value is simply a term that
never reaches WHNF — divergence IS the model's notion of "no truth value".
"""
from __future__ import annotations
from .terms import (Term, Var, Lam, App, Quote, Eval, Fix, Nat,
                     Prim, PartialPrim, Cons)
from .env import Env
from .ops import substitute
from .combinators import Y_COMBINATOR, Z_COMBINATOR, TRUE, FALSE

_MAX_FIX = 5000


def eval_term(term: Term, env: Env, _fd: int = 0) -> Term:
    """
    Evaluate to weak head normal form.

    Primitives (strict):
      car, cdr  — Cons destructuring
      eq        — term equality → Church bool
      add, sub, mult — Nat arithmetic
      iszero    — Nat(0) → TRUE, Nat(n) → FALSE

    Core (lazy):
      (λx.t) s  →  t[x:=s]
      fix(f)    →  f(fix(f))
      eval('t,x) →  t x
    """
    while True:
        match term:
            case Var(name):
                term = env.lookup(name)

            case Lam(_, _) | Nat(_) | Cons(_, _) | Quote(_) | PartialPrim(_, _):
                return term

            case App(func, arg):
                func_val = eval_term(func, env, _fd)

                match func_val:
                    # ── λ-application ──
                    case Lam(param, body):
                        term = substitute(body, param, arg)

                    # ═══════════════════════════════════════════════════════
                    # Nat arithmetic
                    # ═══════════════════════════════════════════════════════
                    case Prim("add"):  term = PartialPrim("add", arg)
                    case Prim("sub"):  term = PartialPrim("sub", arg)
                    case Prim("mult"): term = PartialPrim("mult", arg)
                    case Prim("div"):  term = PartialPrim("div", arg)
                    case Prim("mod"):  term = PartialPrim("mod", arg)
                    case Prim("pow"):  term = PartialPrim("pow", arg)
                    case Prim("min"):  term = PartialPrim("min", arg)
                    case Prim("max"):  term = PartialPrim("max", arg)
                    case Prim("le"):   term = PartialPrim("le", arg)
                    case Prim("lt"):   term = PartialPrim("lt", arg)
                    case Prim("ge"):   term = PartialPrim("ge", arg)
                    case Prim("gt"):   term = PartialPrim("gt", arg)
                    case Prim("pred"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Nat(v): term = Nat(max(0, v - 1))
                            case _: raise ValueError(f"pred expects Nat, got {av}")

                    case PartialPrim("add", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = Nat(v1 + v2)
                            case _: raise ValueError(f"add expects (Nat, Nat)")
                    case PartialPrim("sub", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = Nat(max(0, v1 - v2))
                            case _: raise ValueError(f"sub expects (Nat, Nat)")
                    case PartialPrim("mult", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = Nat(v1 * v2)
                            case _: raise ValueError(f"mult expects (Nat, Nat)")
                    case PartialPrim("div", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = Nat(v1 // v2 if v2 != 0 else 0)
                            case _: raise ValueError(f"div expects (Nat, Nat)")
                    case PartialPrim("mod", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = Nat(v1 % v2 if v2 != 0 else 0)
                            case _: raise ValueError(f"mod expects (Nat, Nat)")
                    case PartialPrim("pow", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)):
                                if v2 > 20:
                                    raise ValueError("pow exponent too large")
                                term = Nat(v1 ** v2)
                            case _: raise ValueError(f"pow expects (Nat, Nat)")
                    case PartialPrim("min", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = Nat(min(v1, v2))
                            case _: raise ValueError(f"min expects (Nat, Nat)")
                    case PartialPrim("max", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = Nat(max(v1, v2))
                            case _: raise ValueError(f"max expects (Nat, Nat)")
                    case PartialPrim("le", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = TRUE if v1 <= v2 else FALSE
                            case _: raise ValueError(f"le expects (Nat, Nat)")
                    case PartialPrim("lt", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = TRUE if v1 < v2 else FALSE
                            case _: raise ValueError(f"lt expects (Nat, Nat)")
                    case PartialPrim("ge", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = TRUE if v1 >= v2 else FALSE
                            case _: raise ValueError(f"ge expects (Nat, Nat)")
                    case PartialPrim("gt", stored):
                        sv, av = eval_term(stored, env, _fd), eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Nat(v1), Nat(v2)): term = TRUE if v1 > v2 else FALSE
                            case _: raise ValueError(f"gt expects (Nat, Nat)")

                    # ═══════════════════════════════════════════════════════
                    # Comparison / Cons / introspection / construction
                    # ═══════════════════════════════════════════════════════
                    case Prim("iszero"):
                        arg_val = eval_term(arg, env, _fd)
                        match arg_val:
                            case Nat(0): term = TRUE
                            case Nat(_): term = FALSE
                            case _: raise ValueError(f"iszero expects Nat, got {arg_val}")

                    case Prim("eq"):
                        term = PartialPrim("eq", arg)
                    case Prim("eq_nat"):
                        term = PartialPrim("eq_nat", arg)

                    case PartialPrim("eq_nat", stored):
                        arg_val, stored_val = eval_term(arg, env, _fd), eval_term(stored, env, _fd)
                        match (stored_val, arg_val):
                            case (Nat(v1), Nat(v2)):
                                term = TRUE if v1 == v2 else FALSE
                            case _:
                                raise ValueError(f"eq_nat expects (Nat, Nat)")

                    case PartialPrim("eq", stored):
                        arg_val, stored_val = eval_term(arg, env, _fd), eval_term(stored, env, _fd)
                        term = TRUE if stored_val == arg_val else FALSE

                    # ── Cons operations ──
                    case Prim("car"):
                        arg_val = eval_term(arg, env, _fd)
                        match arg_val:
                            case Cons(car, _): term = eval_term(car, env)
                            case _: raise ValueError(f"car expects Cons, got {arg_val}")

                    case Prim("cdr"):
                        arg_val = eval_term(arg, env, _fd)
                        match arg_val:
                            case Cons(_, cdr): term = eval_term(cdr, env)
                            case _: raise ValueError(f"cdr expects Cons, got {arg_val}")

                    # ── Term introspection ──
                    case Prim("is_var"):
                        av = eval_term(arg, env, _fd)
                        term = TRUE if isinstance(av, Quote) and isinstance(av.term, Var) else FALSE
                    case Prim("is_lam"):
                        av = eval_term(arg, env, _fd)
                        term = TRUE if isinstance(av, Quote) and isinstance(av.term, Lam) else FALSE
                    case Prim("is_app"):
                        av = eval_term(arg, env, _fd)
                        term = TRUE if isinstance(av, Quote) and isinstance(av.term, App) else FALSE
                    case Prim("is_nat"):
                        av = eval_term(arg, env, _fd)
                        term = TRUE if isinstance(av, Quote) and isinstance(av.term, Nat) else FALSE
                    case Prim("is_quote"):
                        av = eval_term(arg, env, _fd)
                        term = TRUE if isinstance(av, Quote) and isinstance(av.term, Quote) else FALSE
                    case Prim("is_eval"):
                        av = eval_term(arg, env, _fd)
                        term = TRUE if isinstance(av, Quote) and isinstance(av.term, Eval) else FALSE
                    case Prim("is_fix"):
                        av = eval_term(arg, env, _fd)
                        term = TRUE if isinstance(av, Quote) and isinstance(av.term, Fix) else FALSE
                    case Prim("is_cons"):
                        av = eval_term(arg, env, _fd)
                        term = TRUE if isinstance(av, Quote) and isinstance(av.term, Cons) else FALSE

                    case Prim("get_body"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(Lam(_, body)): term = Quote(body)
                            case _: raise ValueError(f"get_body expects quoted Lam")
                    case Prim("get_func"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(App(func, _)): term = Quote(func)
                            case _: raise ValueError(f"get_func expects quoted App")
                    case Prim("get_arg"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(App(_, a)): term = Quote(a)
                            case _: raise ValueError(f"get_arg expects quoted App")
                    case Prim("get_quoted"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(Quote(inner)): term = Quote(inner)
                            case Quote(Eval(q, _)):   term = Quote(q)
                            case _: raise ValueError(f"get_quoted expects quoted Quote/Eval")
                    case Prim("get_eval_arg"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(Eval(_, a)): term = Quote(a)
                            case _: raise ValueError(f"get_eval_arg expects quoted Eval")
                    case Prim("get_fix_func"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(Fix(f)): term = Quote(f)
                            case _: raise ValueError(f"get_fix_func expects quoted Fix")
                    case Prim("get_car"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(Cons(car, _)): term = Quote(car)
                            case _: raise ValueError(f"get_car expects quoted Cons")
                    case Prim("get_cdr"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(Cons(_, cdr)): term = Quote(cdr)
                            case _: raise ValueError(f"get_cdr expects quoted Cons")

                    # ── Term construction ──
                    case Prim("mk_nat"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Nat(v): term = Quote(Nat(v))
                            case _: raise ValueError(f"mk_nat expects Nat, got {av}")
                    case Prim("mk_lam"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Quote(body): term = Quote(Lam("x", body))
                            case _: raise ValueError(f"mk_lam expects quoted body")
                    case Prim("mk_app"):
                        term = PartialPrim("mk_app", arg)
                    case PartialPrim("mk_app", stored):
                        av = eval_term(arg, env, _fd)
                        sv = eval_term(stored, env, _fd)
                        match (sv, av):
                            case (Quote(func), Quote(a)): term = Quote(App(func, a))
                            case _: raise ValueError(f"mk_app expects (quoted func, quoted arg)")

                    # ── Self-reference ──
                    case Fix(f):
                        if _fd >= _MAX_FIX:
                            raise RuntimeError(f"Fix expansion exceeded {_MAX_FIX} layers")
                        term = App(f, Fix(f)); _fd += 1

                    # ── Default: application cannot reduce further ──
                    case _:
                        return App(func_val, arg)

            case Eval(quoted, arg):
                match quoted:
                    case Quote(inner): term = App(inner, arg)
                    case _: term = Eval(eval_term(quoted, env, _fd), arg)

            case Fix(f):
                if _fd >= _MAX_FIX:
                    raise RuntimeError(f"Fix expansion exceeded {_MAX_FIX} layers")
                term = App(f, Fix(f)); _fd += 1

            case _:
                return term


# ═══════════════════════════════════════════════════════════════════════════════
# Default environment
# ═══════════════════════════════════════════════════════════════════════════════

def default_env() -> Env:
    return Env({
        # Nat arithmetic
        "add": Prim("add"), "sub": Prim("sub"), "mult": Prim("mult"),
        "div": Prim("div"), "mod": Prim("mod"), "pow": Prim("pow"),
        "min": Prim("min"), "max": Prim("max"),
        "le": Prim("le"), "lt": Prim("lt"), "ge": Prim("ge"), "gt": Prim("gt"),
        "pred": Prim("pred"),
        "iszero": Prim("iszero"), "eq": Prim("eq"), "eq_nat": Prim("eq_nat"),
        # Cons
        "car": Prim("car"), "cdr": Prim("cdr"),
        # Term introspection
        "is_var": Prim("is_var"), "is_lam": Prim("is_lam"),
        "is_app": Prim("is_app"), "is_nat": Prim("is_nat"),
        "is_quote": Prim("is_quote"), "is_eval": Prim("is_eval"),
        "is_fix": Prim("is_fix"), "is_cons": Prim("is_cons"),
        "get_body": Prim("get_body"), "get_func": Prim("get_func"),
        "get_arg": Prim("get_arg"), "get_quoted": Prim("get_quoted"),
        "get_eval_arg": Prim("get_eval_arg"), "get_fix_func": Prim("get_fix_func"),
        "get_car": Prim("get_car"), "get_cdr": Prim("get_cdr"),
        # Term construction
        "mk_nat": Prim("mk_nat"),
        "mk_lam": Prim("mk_lam"), "mk_app": Prim("mk_app"),
        # Combinators
        "Y": Y_COMBINATOR, "Z": Z_COMBINATOR,
        "True": TRUE, "False": FALSE,
    })


def make_env() -> Env:
    """Full runtime environment. Currently identical to default_env();
    exists as the canonical entry point so callers don't depend on internals.
    """
    return default_env()


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: encode/decode lists
# ═══════════════════════════════════════════════════════════════════════════════

def encode_list(values: list[int]) -> Term:
    """Python list of ints → Cons(Nat, ...) chain."""
    result: Term = Nat(0)
    for v in reversed(values):
        result = Cons(Nat(v), result)
    return result


def decode_nat(term: Term, env: Env) -> int:
    result = eval_term(term, env)
    if isinstance(result, Nat):
        return result.value
    raise ValueError(f"Expected Nat, got {result}")


def decode_list(term: Term, env: Env) -> list[int]:
    result = eval_term(term, env)
    values: list[int] = []
    while isinstance(result, Cons):
        car_val = eval_term(result.car, env)
        if isinstance(car_val, Nat):
            values.append(car_val.value)
        result = eval_term(result.cdr, env)
    return values

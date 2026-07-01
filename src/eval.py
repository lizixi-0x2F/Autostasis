"""Core evaluator — evaluates terms to weak head normal form.

Unified Term space evaluation
  - Fun (scalar = constant function, function = variable function): arithmetic + elementary functions
  - Pointwise algebraic lifting: addf/subf/mulf auto-broadcast over Fun/Lam
  - Function-space primitives: integ (IVP Volterra + BVP Fredholm unified), diff
  - Fun application: App(Fun, Fun) → Fun (function composition)
"""
from __future__ import annotations
import math
import numpy as np
from .terms import (Term, Var, Lam, App, Quote, Eval, Fix, Nat, Domain, Fun,
                     Function,
                     Prim, PartialPrim, Cons, Integ, Diff, Space, SPACE_R, SPACE_C0)
from .env import Env
from .ops import substitute
from .combinators import Y_COMBINATOR, Z_COMBINATOR, TRUE, FALSE

_MAX_FIX = 5000


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical backend
# ═══════════════════════════════════════════════════════════════════════════════

def _trapz(ys, xs):
    """Trapezoidal rule integration."""
    total = 0.0
    for i in range(len(xs) - 1):
        total += (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) / 2.0
    return total


def _simpson(fn, a, b, N=80):
    """∫_aᵇ f(t) dt via Simpson."""
    if abs(b - a) < 1e-15:
        return 0.0
    xs = np.linspace(a, b, N)
    ys = np.array([fn(x) for x in xs])
    return float(_trapz(ys, xs))


def _numerical_derivative(f: Fun, h=1e-5) -> Fun:
    """Central difference numerical derivative."""
    fn = f.fn
    return Fun(lambda x: (fn(x + h) - fn(x - h)) / (2 * h),
               space=f.space, label=f"{f.label}′")


def _numerical_integral(f: Fun, a: float) -> Fun:
    """F(x) = ∫_aˣ f(t) dt."""
    fn = f.fn
    def F(x, _f=fn, _a=a, _N=80):
        return _simpson(_f, _a, x, _N)
    return Fun(F, space=f.space, label=f"∫{f.label}")


def _green_integral(f: Fun, a: float, b: float, N=100) -> Fun:
    """F(x) = ∫_aᵇ G(x,t) f(t) dt, Green's function for −u''=f, u(a)=u(b)=0."""
    fn = f.fn
    def F(x, _f=fn, _a=a, _b=b, _N=N):
        ts = np.linspace(_a, _b, _N)
        f_vals = np.array([_f(t) for t in ts])
        Gx = np.where(ts <= x,
                      (ts - _a) * (_b - x) / (_b - _a),
                      (x - _a) * (_b - ts) / (_b - _a))
        return float(_trapz(Gx * f_vals, ts))
    return Fun(F, space=SPACE_C0(a, b), label=f"G[f]")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: type conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_float(t: Term) -> float:
    """Extract float from a scalar Term.

    A scalar is a Fun on SPACE_R (constant function) which takes the same value at any point.
    """
    match t:
        case Fun() as f: return f(0)
        case Nat(v):     return float(v)
    raise ValueError(f"Cannot extract float from {type(t).__name__}: {t}")


# ═══════════════════════════════════════════════════════════════════════════════
# Integral evaluation — numerical interpretation of Integ terms
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_integ(func: Term, a: Term, b: Term | None, x_val: Term, env: Env) -> Term:
    """App(Integ(func, a, b), x_val) → integral value at x_val (Fun scalar)."""
    # Evaluate bounds
    a_val = eval_term(a, env, 0)
    a_float = _extract_float(a_val)
    if b is not None:
        b_val = eval_term(b, env, 0)
        b_float = _extract_float(b_val)
    x_float = _extract_float(x_val)

    # Evaluate integrand
    f_term = eval_term(func, env, 0)

    if b is None:
        # ── IVP: ∫_aˣ f(t) dt  ──
        match f_term:
            case Fun() as f:
                integral_f = _numerical_integral(f, a_float)
            case Function() as fn:
                f = Fun(lambda t, _f=fn: _f.eval_at(t, env),
                        space=SPACE_C0(a_float, a_float + 10.0))
                integral_f = _numerical_integral(f, a_float)
            case _:
                raise ValueError(f"Integ IVP needs Fun or Lam integrand, got {type(f_term).__name__}")
        # Evaluate the integral function at point x
        return Fun(lambda _, v=float(integral_f(x_float)): v, space=SPACE_R)
    else:
        # ── BVP: ∫_aᵇ G(x,t) f(t) dt  ──
        match f_term:
            case Fun() as f:
                integral_f = _green_integral(f, a_float, b_float)
            case Function() as fn:
                f = Fun(lambda t, _f=fn: _f.eval_at(t, env),
                        space=SPACE_C0(a_float, b_float))
                integral_f = _green_integral(f, a_float, b_float)
            case _:
                raise ValueError(f"Integ BVP needs Fun or Lam integrand, got{type(f_term).__name__}")
        return Fun(lambda _, v=float(integral_f(x_float)): v, space=SPACE_R)


def _eval_diff(func: Term, x_val: Term, env: Env, h: float = 1e-5) -> Term:
    """App(Diff(func), x_val) -> f'(x) via central finite difference.

    Fun:   numerical derivative of a Python callable
    Lam:   symbolic derivative via finite difference on the body
           ∂(λp.body)/∂p at p=x_val ≈ (body[p:=x+h] - body[p:=x-h]) / 2h
    """
    x_float = _extract_float(x_val)
    f_term = eval_term(func, env, 0)
    match f_term:
        case Fun() as f:
            deriv = _numerical_derivative(f)
            return Fun(lambda _, v=float(deriv(x_float)): v, space=SPACE_R)
        case Function() as fn:
            # ∂f/∂x at x_val via central finite difference on eval_at
            f_plus = fn.eval_at(x_float + h, env)
            f_minus = fn.eval_at(x_float - h, env)
            deriv_val = (f_plus - f_minus) / (2 * h)
            return Fun(lambda _, v=deriv_val: v, space=SPACE_R)
        case _:
            raise ValueError(f"Diff requires Fun or Lam, got {type(f_term).__name__}")


# ═══════════════════════════════════════════════════════════════════════════════
# Core evaluator
# ═══════════════════════════════════════════════════════════════════════════════

def eval_term(term: Term, env: Env, _fd: int = 0) -> Term:
    """
    Evaluate to weak head normal form.

    Primitives (strict):
      car, cdr  — Cons destructuring
      eq        — term equality → Church bool
      add, sub, mult — Nat arithmetic
      iszero    — Nat(0) → TRUE, Nat(n) → FALSE

    Arithmetic (unified Term space):
      addf, subf, mulf, divf, powf, negf
      sqrtf, expf, logf, sinf, cosf

    Pointwise algebraic lifting:
      addf/subf/mulf (Fun|Lam) × (Fun|Lam) → auto-broadcast

    Function-space primitives:
      integ(f, a)  → Fun: F(x) = ∫_aˣ f(t) dt
      diff(f)      → Fun: f'
      integ(f, Cons(a, Cons(b, _))) → Fun: ∫ G(x,t) f(t) dt  (BVP)

    Core (lazy/cbv):
      (λx.t) s  →  t[x:=s]
      fix(f)    →  f(fix(f))
      eval('t,x) →  t x
    """
    while True:
        match term:
            case Var(name):
                term = env.lookup(name)

            case Lam(_, _) | Nat(_) | Cons(_, _) | Quote(_) | Fun(_) | PartialPrim(_, _) | Integ(_, _, _) | Diff(_):
                return term

            case App(func, arg):
                func_val = eval_term(func, env, _fd)

                match func_val:
                    # ── λ-application ──
                    case Lam(param, body):
                        term = substitute(body, param, arg)

                    # ═══════════════════════════════════════════════════════
                    # Nat arithmetic (original)
                    # ═══════════════════════════════════════════════════════
                    case Prim("add"): term = PartialPrim("add", arg)
                    case Prim("sub"): term = PartialPrim("sub", arg)
                    case Prim("mult"): term = PartialPrim("mult", arg)

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

                    # ═══════════════════════════════════════════════════════
                    # Fun arithmetic
                    # ═══════════════════════════════════════════════════════
                    case Prim("addf"):  term = PartialPrim("addf", arg)
                    case Prim("subf"):  term = PartialPrim("subf", arg)
                    case Prim("mulf"):  term = PartialPrim("mulf", arg)
                    case Prim("divf"):  term = PartialPrim("divf", arg)
                    case Prim("powf"):  term = PartialPrim("powf", arg)
                    case Prim("negf"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Fun() as f:
                                term = Fun(lambda x, _f=f: -_f(x),
                                           space=f.space)
                            case Lam(p, b):
                                term = Lam(p, App(Prim("negf"), b))
                            case _:
                                return App(Prim("negf"), av)

                    # addf / subf / mulf — unified pointwise algebra
                    case PartialPrim("addf", stored):
                        term = _binop(lambda a, b: a + b, "addf",
                                      stored, arg, env, _fd)
                    case PartialPrim("subf", stored):
                        term = _binop(lambda a, b: a - b, "subf",
                                      stored, arg, env, _fd)
                    case PartialPrim("mulf", stored):
                        term = _binop(lambda a, b: a * b, "mulf",
                                      stored, arg, env, _fd)
                    case PartialPrim("divf", stored):
                        sv = eval_term(stored, env, _fd)
                        av = eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Function() as f, Function() as g):
                                # Symbolic Lam×Lam with matching param: push into body
                                if isinstance(f, Lam) and isinstance(g, Lam) and f.param == g.param:
                                    term = Lam(f.param, App(App(Prim("divf"), f.body), g.body))
                                else:
                                    f_num = _to_numerical(f, g.space, env)
                                    g_num = _to_numerical(g, f.space, env)
                                    term = Fun(lambda x, _f=f_num, _g=g_num:
                                               _f(x) / _g(x) if _g(x) != 0 else float("inf"),
                                               space=f.space)
                            case _: return App(App(Prim("divf"), sv), av)
                    case PartialPrim("powf", stored):
                        sv = eval_term(stored, env, _fd)
                        av = eval_term(arg, env, _fd)
                        match (sv, av):
                            case (Fun() as f, Fun() as g):
                                try:
                                    term = Fun(lambda x, _f=f, _g=g: _f(x) ** _g(x),
                                               space=f.space)
                                except OverflowError:
                                    term = Fun(lambda _: float("inf"), space=f.space)
                            case _: return App(App(Prim("powf"), sv), av)

                    # ═══════════════════════════════════════════════════════
                    # Elementary functions
                    # ═══════════════════════════════════════════════════════
                    case Prim("sqrtf"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Fun() as f:
                                term = Fun(lambda x, _f=f: math.sqrt(_f(x)),
                                           space=f.space)
                            case _: return App(Prim("sqrtf"), av)
                    case Prim("expf"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Fun() as f:
                                term = Fun(lambda x, _f=f: math.exp(_f(x)),
                                           space=f.space)
                            case _: return App(Prim("expf"), av)
                    case Prim("logf"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Fun() as f:
                                term = Fun(lambda x, _f=f: math.log(_f(x)),
                                           space=f.space)
                            case _: return App(Prim("logf"), av)
                    case Prim("sinf"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Fun() as f:
                                term = Fun(lambda x, _f=f: math.sin(_f(x)),
                                           space=f.space)
                            case _: return App(Prim("sinf"), av)
                    case Prim("cosf"):
                        av = eval_term(arg, env, _fd)
                        match av:
                            case Fun() as f:
                                term = Fun(lambda x, _f=f: math.cos(_f(x)),
                                           space=f.space)
                            case _: return App(Prim("cosf"), av)

                    # ═══════════════════════════════════════════════════════
                    # Fun function application: Fun(Fun) → Fun (function composition f∘g)
                    # ═══════════════════════════════════════════════════════
                    case Fun() as f:
                        arg_val = eval_term(arg, env, _fd)
                        match arg_val:
                            case Fun() as g:
                                try:
                                    term = Fun(lambda x, _f=f, _g=g: _f(_g(x)),
                                               space=f.space)
                                except Exception:
                                    term = Fun(lambda _: float("nan"), space=f.space)
                            case _:
                                return App(f, arg_val)

                    # ========================================================
                    # Function-space operators: Integ, Diff (first-class Terms)
                    # ========================================================
                    case Integ(func, a, b):
                        x_val = eval_term(arg, env, _fd)
                        match _eval_integ(func, a, b, x_val, env):
                            case Fun() as f: term = f
                            case other: return other

                    case Diff(func):
                        x_val = eval_term(arg, env, _fd)
                        term = _eval_diff(func, x_val, env)

                    # ═══════════════════════════════════════════════════════
                    # Comparison / Cons / introspection / construction (original)
                    # ═══════════════════════════════════════════════════════
                    case Prim("iszero"):
                        arg_val = eval_term(arg, env, _fd)
                        match arg_val:
                            case Nat(0): term = TRUE
                            case Nat(_): term = FALSE
                            case Fun() as f:
                                term = TRUE if f(0) == 0 else FALSE
                            case _: raise ValueError(f"iszero expects Nat/Fun, got {arg_val}")

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
# Lam → Fun compilation (single-shot sampling, eliminates per-point eval_term)
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_lam(lam: Function, space: Space, env: Env, n: int = 100) -> Fun:
    """Compile a symbolic Function (Lam) into a numerical Fun (interpolated sampling)."""
    dom = space.domain or Domain(0.0, 1.0)
    xs = np.linspace(dom.a, dom.b, n)
    ys = np.array([_extract_float(
        eval_term(App(lam, Fun(lambda _, v=float(x): v)), env, 0)) for x in xs])
    return Fun(lambda x, _xs=xs, _ys=ys: float(np.interp(x, _xs, _ys)),
               space=Space("C⁰", domain=dom))


# ═══════════════════════════════════════════════════════════════════════════════
# Function conversion helper
# ═══════════════════════════════════════════════════════════════════════════════

def _to_numerical(f: Function, target_space: Space, env: Env) -> Fun:
    """Convert a Function to a numerical Fun. Fun passes through; Lam compiles."""
    if isinstance(f, Fun):
        return f
    return _compile_lam(f, target_space, env)


# ═══════════════════════════════════════════════════════════════════════════════
# Pointwise binary operations (addf / subf / mulf)
# ═══════════════════════════════════════════════════════════════════════════════

def _binop(op, op_name, stored, arg, env, _fd):
    """Pointwise binary operation with unified Function broadcasting.

    (Function, Function) with matching Lam params → symbolic push.
    Otherwise compile Lam → Fun, then pointwise Fun×Fun.
    """
    sv = eval_term(stored, env, _fd)
    av = eval_term(arg, env, _fd)
    match (sv, av):
        case (Function() as f, Function() as g):
            # Symbolic: push into body when both are Lam with same param
            if isinstance(f, Lam) and isinstance(g, Lam) and f.param == g.param:
                return Lam(f.param, App(App(Prim(op_name), f.body), g.body))
            # Cross-boundary or both Fun: compile Lam → Fun, then pointwise
            f_num = _to_numerical(f, g.space, env)
            g_num = _to_numerical(g, f.space, env)
            return Fun(lambda x, _f=f_num, _g=g_num: op(_f(x), _g(x)),
                       space=f.space)
        case _:
            return App(App(Prim(op_name), sv), av)


# ═══════════════════════════════════════════════════════════════════════════════
# Default environment
# ═══════════════════════════════════════════════════════════════════════════════

def default_env() -> Env:
    return Env({
        # Nat arithmetic
        "add": Prim("add"), "sub": Prim("sub"), "mult": Prim("mult"),
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
        "mk_lam": Prim("mk_lam"), "mk_app": Prim("mk_app"),
        # Combinators
        "Y": Y_COMBINATOR, "Z": Z_COMBINATOR,
        "True": TRUE, "False": FALSE,
        # ── Unified Term space: real arithmetic + elementary functions ──
        "addf": Prim("addf"), "subf": Prim("subf"),
        "mulf": Prim("mulf"), "divf": Prim("divf"),
        "powf": Prim("powf"), "negf": Prim("negf"),
        "sqrtf": Prim("sqrtf"), "expf": Prim("expf"),
        "logf": Prim("logf"), "sinf": Prim("sinf"),
        "cosf": Prim("cosf"),
        # Note: integ and diff are now first-class Terms (Integ, Diff),
        # not Prims. They don't need env entries.
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime environment
# ═══════════════════════════════════════════════════════════════════════════════

def make_env() -> Env:
    """Full runtime environment. Currently identical to default_env();
    exists as the canonical entry point so callers don't depend on internals.
    """
    return default_env()


# ═══════════════════════════════════════════════════════════════════════════════
# Term → flat Fun compilation — bridge from symbolic to numerical
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_term(term: Term, sp: Space, n: int = 100) -> Fun:
    """Compile any Term into a flat Fun without closure chains.

    Scalar space: extract numeric value, return constant function.
    Function space: sample over domain → np.interp interpolation.
    This is the infrastructure for numerical evaluation — the bridge from symbolic Term to numerical Fun.
    """
    if sp.domain is None:
        return Fun(lambda _, v=_extract_float(term): v, space=sp)
    xs = np.linspace(sp.domain.a, sp.domain.b, n)
    ys = np.array([float(term(x)) for x in xs])
    return Fun(lambda x, _xs=xs, _ys=ys: float(np.interp(x, _xs, _ys)),
               space=Space("C⁰", domain=sp.domain))


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

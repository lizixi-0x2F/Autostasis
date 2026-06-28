"""
Unified equation (D, C)u = (f, g)
=================================
  u in X      unknown object
  Du = f      interior law
  Cu = g      exterior constraint

Unique transformation: T_eta[u] = u - eta * A[u]
Formal solution:       Fix(T) -- fixed point IS the solution

IVP  = C takes initial slice
BVP  = C takes boundary slice
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.terms import Var, Lam, App, Prim, Fun, SPACE_C0, space_of
from src.eval import eval_term, make_env, flatten_term
from src.dsl import R, add, sub, mul, pow_, integ
from src.solve import solve


# =====================================================================
# Evaluation harness
# =====================================================================

def show(sol, x0, *, precision=1e-8, max_iter=200, label="solve"):
    """Numerical port: Fix(T) + x0 -> iterate until convergence."""
    from src.terms import Fix
    match sol:
        case Fix(T): pass
        case _: raise ValueError("sol must be Fix(T)")

    env = make_env()
    sp = space_of(x0)
    x = x0

    print(f"\n{'-' * 40}")
    print(f"  {label}")
    print(f"  space: {sp}")
    if sp.domain is not None:
        print(f"  init:  {x.sample()}")
    else:
        print(f"  init:  x0 = {x}")
    print(f"  {'iter':>4s}  {'|delta|':>14s}")

    for i in range(max_iter):
        result = eval_term(App(T, x), env)
        x_new = flatten_term(result, sp)
        change = sp.distance(x_new, x, env)

        if i < 5 or change < precision * 100 or i % 50 == 0:
            print(f"  {i:>4d}  {change:>14.4e}")

        if math.isnan(change) or math.isinf(change):
            print(f"  X diverged"); return None
        if change < precision:
            print(f"  V converged at iter {i + 1}")
            return x_new
        x = x_new

    print(f"  ! max_iter={max_iter} reached")
    return x


# =====================================================================
# Demo: scalar -> IVP -> BVP, one pipeline
# =====================================================================

if __name__ == "__main__":
    x = Var("x")
    y = Var("y")
    u = Var("u")
    t = Var("t")

    # -- Scalar: x^3 - 2x - 5 = 0 --
    A1 = Lam("x", sub(sub(pow_(x, R(3.0)), mul(R(2.0), x)), R(5.0)))
    x1 = show(solve(A1, eta=0.1), R(2.0), label="x^3 - 2x - 5 = 0")
    if x1:
        print(f"  x* = {x1(0):.12f},  f(x*) = {x1(0)**3 - 2*x1(0) - 5:.1e}")

    # -- Scalar: x = cos(x) --
    A2 = Lam("x", sub(x, App(Prim("cosf"), x)))
    x2 = show(solve(A2, eta=1.0), R(0.5), label="x = cos(x), eta=1 -> T(x)=cos(x)")
    if x2:
        v = x2(0)
        print(f"  Fix(cos) = {v:.12f},  cos(x*) = {math.cos(v):.12f}")

    # -- IVP: y' = y, y(0)=1 -> y(x)=exp(x) --
    # A[y](x) = y(x) - 1 - int_0^x y(t) dt
    C0 = SPACE_C0(0.0, 1.0)
    y0 = Fun(lambda _: 1.0, space=C0, label="y0=1")

    A_ivp = Lam("y", Lam("x",
        sub(App(y, x), add(R(1.0),
            App(integ(Lam("t", App(y, t)), R(0.0)), x)))))
    ys = show(solve(A_ivp, eta=1.0), y0, label="y'=y, y(0)=1, eta=1 (Picard)")
    if ys:
        exact = Fun(math.exp, space=C0)
        err = C0.distance(ys, exact)
        print(f"  ||y - exp||_L2 = {err:.2e}")

    # -- BVP: -u'' = 1, u(0)=u(1)=0 -> u(x)=x(1-x)/2 --
    # A[u](x) = u(x) - 0 - G[1](x)
    L = 1.0
    C0_bvp = SPACE_C0(0.0, L)
    u0 = Fun(lambda _: 0.0, space=C0_bvp, label="u0=0")

    A_bvp = Lam("u", Lam("x",
        sub(App(u, x),
            App(integ(Lam("t", R(1.0)), R(0.0), R(L)), x))))
    us = show(solve(A_bvp, eta=0.5), u0, label="-u''=1, u(0)=u(1)=0", max_iter=300)
    if us:
        exact_bvp = Fun(lambda x: x * (1 - x) / 2, space=C0_bvp)
        err = C0_bvp.distance(us, exact_bvp)
        print(f"  ||u - x(1-x)/2||_L2 = {err:.2e}")

    print(f"\n{'=' * 40}")
    print(f"  (D, C)u = (f, g)  ->  T  ->  Fix(T)  =  solution")
    print(f"{'=' * 40}\n")

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

from src.terms import Var, Lam, App, Prim, Fun, Fix, SPACE_C0, space_of
from src.eval import make_env, fixpoint
from src.dsl import R, add, sub, mul, pow_, integ
from src.solve import solve


# =====================================================================
# Evaluation harness
# =====================================================================

def show(sol, x0, *, precision=1e-8, max_iter=200, label="solve"):
    """Numerical port: Fix(T) + x0 -> iterate until convergence.

    The loop lives in eval.fixpoint — show() is only the display shell.
    """
    match sol:
        case Fix(_): pass
        case _: raise ValueError("sol must be Fix(T)")

    env = make_env()
    sp = space_of(x0)

    print(f"\n{'-' * 40}")
    print(f"  {label}")
    print(f"  space: {sp}")
    if sp.domain is not None:
        print(f"  init:  {x0.sample()}")
    else:
        print(f"  init:  x0 = {x0}")
    print(f"  {'iter':>4s}  {'|delta|':>14s}")

    def on_iter(i, x_old, x_new, change):
        if i < 5 or change < precision * 100 or i % 50 == 0:
            print(f"  {i:>4d}  {change:>14.4e}")
        if change < precision:
            print(f"  V converged at iter {i + 1}")

    result = fixpoint(sol, x0, tol=precision, max_iter=max_iter, env=env,
                      on_iter=on_iter)
    if result is None:
        print(f"  X diverged")
    return result


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

    # -- PDE: heat equation u_t = u_xx via implicit Euler + Fix --
    # (I - dt*d^2/dx^2) u^{n+1} = u^n  rewritten via Green's K = (-d^2/dx^2)^{-1}:
    #   u + alpha*K[u] = alpha*K[u^n],  alpha = 1/dt
    #   A[u] = u + alpha*K[u] - alpha*K[u^n] = 0
    dt = 0.1
    alpha = 1.0 / dt  # = 10
    n_steps = 3
    C0 = SPACE_C0(0.0, 1.0)

    # initial condition: u(x,0) = sin(pi*x)
    u_heat = Fun(lambda x: math.sin(math.pi * x), space=C0, label="u0")

    ux = Var("u"); xx = Var("x"); tt = Var("t")
    for step in range(n_steps):
        t_now = dt * (step + 1)
        u_n = u_heat  # freeze current state as source term

        # K[u^n]: Green integral of known state (constant during this step)
        K_un = integ(Lam("t", App(u_n, tt)), R(0.0), R(1.0))

        # A[u](x) = u(x) + alpha*K[u](x) - alpha*K[u^n](x)
        A_heat = Lam("u", Lam("x",
            sub(add(App(ux, xx),
                    mul(R(alpha), App(integ(Lam("t", App(ux, tt)), R(0.0), R(1.0)), xx))),
                mul(R(alpha), App(K_un, xx)))))

        u_heat = show(solve(A_heat, eta=0.5), u_n,
                      label=f"Heat step {step+1}/{n_steps}, t={t_now:.2f}",
                      max_iter=100)
        if u_heat is None: break

    if u_heat:
        # compare vs exact PDE solution u(x,t) = exp(-pi^2*t)*sin(pi*x)
        t_final = dt * n_steps
        exact_heat = Fun(lambda x: math.exp(-math.pi**2 * t_final) * math.sin(math.pi * x),
                         space=C0)
        err = C0.distance(u_heat, exact_heat)
        print(f"  PDE exact vs implicit Euler at t={t_final:.2f}:")
        print(f"    u_num(0.5)  = {u_heat(0.5):.6f}")
        print(f"    u_exact(0.5) = {exact_heat(0.5):.6f}")
        print(f"    ||u - u_exact||_L2 = {err:.2e}")
        print(f"    (O(dt) temporal error, vanishes as dt->0)")

    print(f"\n{'=' * 40}")
    print(f"  (D, C)u = (f, g)  ->  T  ->  Fix(T)  =  solution")
    print(f"{'=' * 40}\n")

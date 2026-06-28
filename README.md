# Autostasis

A computational model where **self-reference and fixed points are the primitive operations**.

Every equation is a functional equation `A[u] = 0`. The transform `T[u] = u - η·A[u]` is canonical. `Fix(T)` is the solution—as a syntactic theorem, not a numerical heuristic.

```
(D, C)u = (f, g)   ->   T   ->   Fix(T)   =   solution
```

IVP is C taking an initial slice. BVP is C taking boundary slices. PDE is BVP repeated in time. One pipeline.

## Quick start

```bash
python experiments/ats_solver.py
```

Requires `numpy`. No other dependencies.

## Structure

```
src/
  terms.py      -- unified Term space: Var..Fix, Fun, Space, Integ, Diff
  eval.py       -- WHNF evaluator, numerical backend, flatten_term
  solve.py      -- T(A, eta) and solve(A, eta) -- 10 lines
  dsl.py        -- construction sugar: R, add, sub, mul, integ, diff_op
  ops.py        -- free_vars, capture-avoiding substitution
  env.py        -- lexical environment
  combinators.py -- Y, Z, Church booleans
  serialize.py  -- JSON, S-expression round-trips
```

## The unified equation

```
u in X        unknown object
Du = f        interior law (differential operator)
Cu = g        exterior constraint (slice operator)
---------------------------------------------------
(D, C)u = (f, g)
```

| C takes | Equation type |
|---------|---------------|
| nothing | scalar equation |
| `u(a)` | IVP |
| `(u(a), u(b))` | BVP |
| `(u(a), u(b))` × N_t | PDE (time-dependent) |

## The T-transform

```python
def T(A, eta=0.1):
    """T[u] = u - eta * A[u]"""
    x = Var("x")
    return Lam("x", sub(x, mul(R(eta), App(A, x))))

def solve(A, eta=0.1):
    """Fix(T(A, eta)) -- fixed point IS the solution.

    Proof:
      Fix(T) = T(Fix(T)) = Fix(T) - eta * A(Fix(T))
      => A(Fix(T)) = 0
    """
    return Fix(T(A, eta))
```

## Demos

| Equation | Type | (D, C) | Fix iters |
|----------|------|--------|------------|
| x³ − 2x − 5 = 0 | scalar | (I, ∅) | 9 |
| x = cos(x) | scalar | (I, ∅) | 45 |
| y′ = y, y(0)=1 | IVP | (d/dx, eval\|₀) | 11 |
| −u″ = 1, u(0)=u(1)=0 | BVP | (−d²/dx², eval\|₀₊₁) | 24 |
| u_t = u_xx, u(x,0)=sin(πx) | PDE (heat) | (−d²/dx², eval\|₀₊₁) × N_t | ~5/step |

### How the PDE works

The heat equation is solved via **implicit Euler time-stepping + Fix per step**:

```
(I - dt*d^2/dx^2) u^{n+1} = u^n
```

Using the Green's function `K = (-d^2/dx^2)^{-1}` (same as the BVP case), rewrite as:

```
u + (1/dt)*K[u] = (1/dt)*K[u^n]
A[u] = u + alpha*K[u] - alpha*K[u^n] = 0
```

Then `Fix(T(A))` solves the linear system at each time step. The PDE is just the BVP operator applied repeatedly in time. Same `Integ`, same `T`, same `Fix`.

## Design

- **Scalar = constant function on `SPACE_R`**. No `Real` vs `Fun` type split. `addf/subf/mulf` broadcast pointwise over everything.
- **`Space` is a first-class `Term`** carrying `distance`, `norm`, `zero`. The solver never branches on space type.
- **`Integ` and `Diff` are Terms**, not Python escape hatches. They can be introspected, substituted, serialized.
- **Form and value are separated**. `solve()` returns a formal `Fix(T)` term. `show()` evaluates its numerical interpretation. Same solution, different evaluation strategies.

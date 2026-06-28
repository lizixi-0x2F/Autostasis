# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Autostasis is a computational model where self-reference and fixed points are primitive operations. Every equation is treated as a functional equation `A[u] = 0`, the Euler step `T[f] = f - eta * A[f]` is the sole transformation, and `Fix(T)` is the solution -- not as a numerical heuristic but as a syntactic theorem: `Fix(T) = T(Fix(T))` implies `A(Fix(T)) = 0`.

The demo lives in `experiments/ats_solver.py`. The core library is `src/`.

## Architecture

### Term space (src/terms.py)

Every inhabitant participates in the same algebra. No `Real` vs `Fun` type split:

| Layer | Types | Role |
|-------|-------|------|
| Computation | `Var`, `Lam`, `App` | lambda-calculus |
| Self-reference | `Quote`, `Eval`, `Fix` | reification, self-application, fixed points |
| Data | `Nat`, `Cons` | natural numbers, pairs |
| Quantity | `Fun`, `Space` | vector-space points and the spaces themselves |
| Operators | `Integ`, `Diff` | integral/derivative as first-class Terms, not Prim escapes |

Key design: **scalar = constant function on `SPACE_R`**. `constant(3.14)` and `Fun(lambda x: sin(x))` are the same type in different spaces. `addf/subf/mulf` broadcast pointwise over both scalars and functions automatically.

`Space` is a first-class `Term` carrying `distance`, `norm`, `zero`, `contains`. The solver never branches on space type -- `sp.distance(a, b)` dispatches to the right metric.

### Module map

| Module | Purpose |
|--------|---------|
| `src/terms.py` | All Term types + Space + Domain + Fun |
| `src/eval.py` | Core evaluator (`eval_term`), numerical backend, `_binop`, `_compile_lam`, `flatten_term`, `make_env` |
| `src/ops.py` | `free_vars`, `substitute` (capture-avoiding, handles all Term types) |
| `src/env.py` | `Env` with chained parent lookup |
| `src/solve.py` | `T(A, eta)` and `solve(A, eta)` -- the solver kernel (10 lines) |
| `src/dsl.py` | Construction sugar: `R()`, `add()`, `sub()`, `mul()`, `pow_()`, `integ()`, `diff_op()` |
| `src/combinators.py` | `Y_COMBINATOR`, `Z_COMBINATOR`, Church `TRUE`/`FALSE` |
| `src/serialize.py` | JSON and S-expression round-trips for all Term types |

### Evaluator (src/eval.py)

`eval_term(term, env, _fd)` evaluates to weak head normal form in a `while True` loop.

Function-space operators as first-class Terms (not `Prim` escapes):
- **`Integ(func, a, b)`**: `App(Integ(...), x)` dispatches via `_eval_integ` to `_numerical_integral` (IVP / Volterra) or `_green_integral` (BVP / Fredholm).
- **`Diff(func)`**: `App(Diff(...), x)` dispatches via `_eval_diff` to `_numerical_derivative`.

`_binop` broadcasts `addf/subf/mulf` over `Fun`/`Lam`:
- `(Fun, Fun)` -> pointwise.
- `(Fun, Lam)` -> **compile Lam to Fun first** (`_compile_lam`), then pointwise. Critical: never push into Lam body during iteration.
- `(Lam, Lam)` with same param -> symbolic push (for construction, not iteration).

`divf` follows the same rules as `_binop` -- `(Fun, Lam)` compiles via `_compile_lam`, never pushes into Lam body.

`_compile_lam`: samples a `Lam` on a domain in one shot, returns a flat `np.interp`-based `Fun` that never triggers `eval_term` again.

### Critical correctness rules

1. **Never push `(Fun, Lam)` into the Lam body** during iteration. This applies to `_binop` AND `divf`. The old code had `(Fun, Lam) -> Lam(p, App(...))` cases that caused Lam nesting to grow O(iterations). Always compile `Lam -> Fun` before combining with `Fun`.

2. **Flatten after each Euler step.** `_binop(Fun, Fun) -> Fun(lambda x: op(f(x), g(x)))` creates closures capturing the previous Fun. After n iterations, `result(x)` chains through n closures. Use `flatten_term` (sample on grid -> `np.interp`) to break the chain.

3. **`_compile_lam` must use a function-space domain**, never compile a function-valued Lam as a scalar (single point). If the space has no domain, default to `[0, 1]`.

4. **`flatten_term` for scalar spaces** extracts the value and wraps in `constant(...)`. For function spaces, sample `term(x)` directly -- do NOT wrap in a dummy Lam (the old `Lam("_", term)` bug caused all samples to evaluate at the same point).

5. **`Integ` and `Diff` are Terms, not Prims.** They don't live in the environment. They are constructed directly as `Integ(func, a, b)` / `Diff(func)` and evaluated when applied to a point.

## Running

```bash
python experiments/ats_solver.py
```

No external dependencies beyond `numpy`. No build step, no test runner.

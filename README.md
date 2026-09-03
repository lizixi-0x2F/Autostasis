# Autostasis

A computational model where **self-reference and fixed points are the primitive operations** — focused on logic.

The Term space is a λ-calculus with first-class quotation:

| Layer | Terms | Role |
|-------|-------|------|
| Computation | `Var`, `Lam`, `App` | λ-calculus |
| Self-reference | `Quote`, `Eval`, `Fix` | a language that talks about itself |
| Data | `Nat`, `Cons` | numbers, pairs |

```python
from src import *

env = make_env()
term = App(App(Prim("add"), Nat(2)), Nat(3))
eval_term(term, env)      # → Nat(5)

# fix(f) → f(fix(f)) — Kleene's recursion theorem, as syntax
```

## The self-reference primitives

- `Quote(t)` — reify a term as data. `'t` in the repr.
- `Eval(q, x)` — execute quoted data: `eval(quote(t), x) → t x`.
- `Fix(f)` — `fix(f) → f(fix(f))`, Kleene's first recursion theorem.

These are what make the model a natural home for logic. Self-reference —
Gödel coding, provability predicates, paradoxes — is usually a chapter of
painful encoding. Here it is free syntax.

## What lives in the model

- **λ-calculus** with capture-avoiding substitution, WHNF evaluation
- **Nat arithmetic**: `add`, `sub` (monus), `mult`, `iszero`, `eq`, `eq_nat`
- **Introspection**: `is_var`, `is_lam`, `get_body`, `get_func`, ... — the
  language can inspect its own terms (the raw material of a proof checker)
- **Construction**: `mk_lam`, `mk_app` — build quoted terms at runtime
- **Combinators**: `Y`, `Z`, Church `True`/`False`
- **Serialization**: JSON and S-expression round-trips for every term

## Why logic is the natural target

- **Proofs are terms, derivations are reductions.** `App(f, x)` *is* modus
  ponens (Curry–Howard): `f` proves `A ⊃ B`, `x` proves `A`, the
  application proves `B`. β-reduction *is* proof normalization
  (Gentzen's cut elimination).
- **Paradoxes are divergent terms.** Curry's paradox — `Fix(λc. imp(c, A))`
  — never reaches WHNF. Divergence is the model's native notion of
  "no truth value": no three-valued logic needs to be bolted on.
- **Truth is a fixed point.** Kripke's theory of truth (1975) defines truth
  as the least fixed point of a monotone operator on truth values — the
  same engine as solving every equation, now on the space of sentences.

## Structure

```
src/
  terms.py      -- unified Term space: Var..Fix, Nat, Cons
  eval.py       -- WHNF evaluator, introspection, default environment
  ops.py        -- free variables, capture-avoiding substitution
  env.py        -- lexical environment
  combinators.py -- Y, Z, Church booleans
  serialize.py  -- JSON, S-expression round-trips
```

Requires no dependencies beyond the standard library.

## Arrow — a synthesis compiler (`python -m arrow.main`)

`arrow spec.arr -o out.c`: **examples in, C out.** A real three-phase
compiler (frontend → synthesis → codegen) built on the Term space:

```
int f(int x)          # spec
f(0) = 1
f(1) = 3
f(2) = 5
```

Arrow enumerates terms bottom-up by increasing size; the first term
consistent with every example is by construction **the shortest one** —
MDL induction, Occam's razor as a compiler pass. The result is
decompiled to readable C (Church booleans become `?:`), and gcc takes
over from there.

Diagnostics follow compiler convention (`arrow: error: file:line: msg`,
exit 2 = parse error, 1 = synthesis failed, 0 = success).

Next step (the self-referential loop): every synthesized program becomes
a primitive for the next search — the search language is rewritten by its
own results, step after step, until the target program is reached.

## The playground

- **Metacircular evaluator** — implement `eval` *inside* the Term space with
  `Quote`/`Eval`/introspection. Then ask: do the inner and outer evaluators
  ever disagree? (reflection, intuitively)
- **Fix vs Y** — `Fix(f) → f(Fix(f))` shares structure (zero copies); `Y`
  expands with copying. Two fixed-point economies; count the reduction steps.
- **Paradox zoo** — one Term per paradox (Curry, Berry via a definability
  predicate, Grelling), classified by evaluation behavior: divergence is
  the model's native "no truth value".
- **Provability predicate** — write `provable(x)` with the introspection
  primitives and test Löb's conditions D1/D2 live. A predicate that
  satisfies them yet is unsound is Gödel II, hands-on.
- **Equational theory of Quote/Eval** — which laws hold? `quote` is
  injective, `eval(quote(t)) → t`; does the syntax/semantics adjunction
  have a precise form here?
- **Internal typechecker** — simply-typed λ as a typing judgment inside
  the model; proofs become typed terms (Curry–Howard, in-model). Then
  watch self-reference bite: quoting one's own type.

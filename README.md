# Autostasis

A computational model where **self-reference and fixed points are the primitive operations** — focused on logic.

The Term space is a λ-calculus with first-class quotation:

| Layer | Terms | Role |
|-------|-------|------|
| Computation | `Var`, `Lam`, `App` | λ-calculus |
| Self-reference | `Quote`, `Eval`, `Fix` | a language that talks about itself |
| Data | `Nat`, `Cons` | numbers, pairs |

```c
/* fix(f) → f(fix(f)) — Kleene's recursion theorem, as syntax */
Term *fib = t_fix(a, t_lam(a, "self", t_lam(a, "x", body)));
Term *r = eval_term(a, t_app(a, fib, t_nat(a, 15)), env, 0);  // Nat(610)
```

## The self-reference primitives

- `Quote(t)` — reify a term as data.
- `Eval(q, x)` — execute quoted data: `eval(quote(t), x) → t x`.
- `Fix(f)` — `fix(f) → f(fix(f))`, Kleene's first recursion theorem.

These are what make the model a natural home for logic. Self-reference —
Gödel coding, provability predicates, paradoxes — is usually a chapter of
painful encoding. Here it is free syntax.

## The core (C)

```sh
make          # gcc -O2
./c/autostasis
```

```
c/term.h      -- Tagged union on an arena: Var..Fix, Nat, Cons, prims,
                 closures, thunks
c/term.c      -- constructors, symbol interning, structural equality
c/eval.c      -- closure-based WHNF evaluator: β binds in the environment
                 (zero tree copying), call-by-name thunks freeze arguments,
                 prims, introspection, construction, Fix expansion
c/main.c      -- benchmarks + fixed cases
```

Terms are a tagged union allocated from a **bump arena** (8 MB blocks, no
per-node malloc). Variable and prim names are **interned** — name
comparison is pointer comparison. Sharing is real: two pointers to one
node are one structure.

**Closure semantics.** β-reduction binds in the environment instead of
rewriting the tree: `App(λx.b, v)` allocates one env node `{x ↦ v}` and
continues under `b` — the retired substitute-based evaluator copied the
whole body at every β. Arguments are frozen as **thunks** under their
defining environment (call-by-name); `Fix` values pass bare and expand
under the use-site environment. **Quote is opaque** (Lisp semantics): the
quoted tree is data and β never touches it — `(λx. 'x) 5 → 'x`.

Lazy WHNF; divergence is guarded by MAX_FIX. Divergence is the model's
native notion of "no truth value" — a term that never reaches WHNF is a
term with no value, not an error to be caught.

Benchmarks (WSL, gcc -O2):

```
fib(15)            0.003s   (the retired Python evaluator: 0.148s)
fib(20)            0.030s
fib(25)            0.39s    (Python: prohibitive)
arith chain 20 000 0.012s
```

The C evaluator was checked against the Python evaluator as an oracle:
22 fixed cases (arith, monus, div-by-zero, Church booleans, Quote/Eval,
introspection, construction, Cons, Fix-recursion) were byte-identical in
s-expression output. The Python `src/` then retired.

## What lives in the model

- **λ-calculus** with capture-avoiding substitution, WHNF evaluation
- **Nat arithmetic**: `add`, `sub` (monus), `mult`, `div`, `mod`, `pow`,
  `min`, `max`, `pred`, `iszero`, comparisons
- **Introspection**: `is_var`, `is_lam`, `get_body`, `get_func`, ... — the
  language can inspect its own terms (the raw material of a proof checker)
- **Construction**: `mk_nat`, `mk_lam`, `mk_app` — build quoted terms at runtime
- **Cons** cells: `car`, `cdr` destructuring

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
- **Monotone program discovery** — a sequence of examples defines a
  monotone operator (sew each sample in); its fixed point is the lookup
  program. Verified: correct-set grows strictly +1 per sample, 120
  orderings → 1 behavior, a single `Fix` term runs the whole loop
  (data in, quoted program out).

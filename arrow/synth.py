"""Synthesis — bottom-up enumeration of the shortest consistent term.

MDL induction: terms are enumerated in increasing size; the first term
consistent with all examples is, by construction, the shortest one.

  search space:  x, 0..3, add, sub (monus), mult, iszero, eq_nat,
                 Church booleans give if-then-else for free

  pruning: behavioral signatures — two terms that behave identically on
  the example inputs collapse; only the shorter survives.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.terms import Var, App, Nat, Prim
from src.combinators import TRUE, FALSE
from src.eval import eval_term, make_env
from src.ops import substitute

ENV = make_env()
X = Var("x")
NAT_CONSTS = [0, 1, 2, 3]


def add(t1, t2):    return App(App(Prim("add"), t1), t2)
def sub(t1, t2):    return App(App(Prim("sub"), t1), t2)
def mult(t1, t2):   return App(App(Prim("mult"), t1), t2)
def iszero(t):      return App(Prim("iszero"), t)
def eq_nat(t1, t2): return App(App(Prim("eq_nat"), t1), t2)
def if_(b, t, e):   return App(App(b, t), e)

# Full primitive spectrum: every binary nat op and every comparison
# the evaluator knows. Behavioral-signature pruning collapses
# observationally equal terms (commutativity, constant folding) — the
# shortest representative survives automatically.
NAT_BINOPS = ["add", "sub", "mult", "div", "mod", "pow", "min", "max"]
BOOL_BINOPS = ["eq_nat", "le", "lt", "ge", "gt"]


def size(t) -> int:
    match t:
        case Var() | Nat():   return 1
        case App(f, a):       return 1 + size(f) + size(a)
        case _:               return 1


def run_nat(t, x: int) -> int | None:
    """Evaluate t with x := Nat(x) substituted in."""
    try:
        r = eval_term(substitute(t, "x", Nat(x)), ENV)
        return r.value if isinstance(r, Nat) else None
    except Exception:
        return None


def run_bool(t, x: int) -> bool | None:
    try:
        r = eval_term(substitute(t, "x", Nat(x)), ENV)
        pick_true = eval_term(App(App(r, Nat(1)), Nat(0)), ENV)
        return isinstance(pick_true, Nat) and pick_true.value == 1
    except Exception:
        return None


class SynthesisError(Exception):
    pass


def synthesize(examples: list[tuple[int, int]], max_size: int = 12,
               verbose: bool = False) -> tuple:
    """Return (term, size, searched) for the shortest consistent term.

    Raises SynthesisError if nothing is found within max_size.
    """
    inputs = [x for x, _ in examples]
    targets = [y for _, y in examples]
    consts = sorted(set(inputs) | set(targets) | {0, 1})

    nat_by_size: dict[int, list] = {}     # size -> [terms]
    bool_by_size: dict[int, list] = {}
    seen_nat: dict = {}                   # signature -> (term, size)
    seen_bool: dict = {}

    def sig_nat(t):
        return tuple(run_nat(t, i) for i in inputs)

    def sig_bool(t):
        return tuple(run_bool(t, i) for i in inputs)

    def add_nat(t, s, sig):
        if sig is None or None in sig:
            return
        if sig not in seen_nat or s < seen_nat[sig][1]:
            seen_nat[sig] = (t, s)
            nat_by_size.setdefault(s, []).append(t)

    def add_bool(t, s, sig):
        if sig is None or None in sig:
            return
        if sig not in seen_bool or s < seen_bool[sig][1]:
            seen_bool[sig] = (t, s)
            bool_by_size.setdefault(s, []).append(t)

    # atoms (size 1)
    add_nat(X, 1, sig_nat(X))
    for c in consts:
        add_nat(Nat(c), 1, sig_nat(Nat(c)))
    add_bool(TRUE, 1, sig_bool(TRUE))
    add_bool(FALSE, 1, sig_bool(FALSE))

    searched = 0

    def check_consistency(s: int):
        for t in nat_by_size.get(s, []):
            outs = [run_nat(t, i) for i in inputs]
            if outs == targets:
                return t
        return None

    found = check_consistency(1)
    if found is not None:
        return found, size(found), searched

    # binary nat ops & eq_nat: size = 3 + s1 + s2
    # iszero: size = 2 + s1
    # if-then-else: size = 2 + sb + st + se
    for s in range(2, max_size + 1):
        # iszero
        for s1 in range(1, s - 1):
            for t1 in nat_by_size.get(s1, []):
                searched += 1
                add_bool(iszero(t1), s, sig_bool(iszero(t1)))
        # binary ops (nat->nat and nat->bool): size = 3 + s1 + s2
        for s1 in range(1, s - 2):
            s2 = s - 3 - s1
            if s2 < 1:
                continue
            for t1 in nat_by_size.get(s1, []):
                for t2 in nat_by_size.get(s2, []):
                    for op in NAT_BINOPS:
                        searched += 1
                        term = App(App(Prim(op), t1), t2)
                        add_nat(term, s, sig_nat(term))
                    for op in BOOL_BINOPS:
                        searched += 1
                        term = App(App(Prim(op), t1), t2)
                        add_bool(term, s, sig_bool(term))
        # if-then-else
        for sb in range(1, s - 2):
            for st in range(1, s - 1 - sb):
                se = s - 2 - sb - st
                if se < 1:
                    continue
                for b in bool_by_size.get(sb, []):
                    for t1 in nat_by_size.get(st, []):
                        for t2 in nat_by_size.get(se, []):
                            searched += 1
                            add_nat(if_(b, t1, t2), s, sig_nat(if_(b, t1, t2)))

        # check consistency at this size
        found = check_consistency(s)
        if found is not None:
            if verbose:
                print(f"arrow: note: size {s} complete ({searched} candidates checked)")
            return found, size(found), searched

        if verbose:
            print(f"arrow: note: size {s}: no consistent term "
                  f"({searched} candidates checked)")

    raise SynthesisError(
        f"no program of size <= {max_size} consistent with the examples")


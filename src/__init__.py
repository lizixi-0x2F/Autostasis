"""Autostasis -- a computational model where self-reference and fixed points
are the primitive operations.

Focused on logic: the Term space is a lambda-calculus with first-class
quotation (Quote), execution of quoted data (Eval), and fixed points
(Fix — Kleene's recursion theorem as syntax). Proofs are terms,
derivations are reductions, paradoxes are divergent terms.
"""
from .terms import (
    Term, Var, Lam, App, Quote, Eval, Fix, Nat,
    Prim, PartialPrim, Cons,
)
from .combinators import Y_COMBINATOR, Z_COMBINATOR, TRUE, FALSE
from .env import Env
from .ops import free_vars, substitute
from .eval import eval_term, default_env, make_env, encode_list, decode_nat, decode_list
from .serialize import (
    term_to_json, term_from_json, term_to_json_bytes, term_from_json_bytes,
    term_to_sexpr,
)

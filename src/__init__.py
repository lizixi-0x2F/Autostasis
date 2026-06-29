"""Autostasis -- a computational model where self-reference and fixed points
are the primitive operations.

Unified Term space: every inhabitant -- Fun (scalar = constant fn,
function = varying fn), Lam, Nat -- participates in the same algebra
(addf, subf, mulf with pointwise lifting).
"""

from .terms import (
    Term, Var, Lam, App, Quote, Eval, Fix, Nat, Domain, Fun,
    Prim, PartialPrim, Cons, Integ, Diff, constant,
)
from .combinators import Y_COMBINATOR, Z_COMBINATOR, TRUE, FALSE
from .env import Env
from .ops import free_vars, substitute
from .eval import eval_term, default_env, make_env, flatten_term, encode_list, decode_nat, decode_list
from .serialize import (
    term_to_json, term_from_json, term_to_json_bytes, term_from_json_bytes,
    term_to_sexpr,
)
from .dsl import R, add, sub, mul, pow_, integ, diff_op
from .solve import T, solve
from .grad import Tape, TapeVar, tv_dot, tv_mse, tv_sum
from .nn import Model, FunctionModel, TapeModel
from .trainer import Trainer

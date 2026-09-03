"""Codegen — decompile a synthesized Term to readable C.

The Term is a pure expression over ints; the C output is a single
function. Nat monus (a ∸ b) becomes a saturating subtraction since C's
'-' is not monus.
"""
from __future__ import annotations
from src.terms import Var, Nat, App, Prim


def _expr(t) -> str:
    match t:
        case Var():
            return "x"
        case Nat(v):
            return str(v)
        case App(App(Prim("add"), a), b):
            return f"({_expr(a)} + {_expr(b)})"
        case App(App(Prim("mult"), a), b):
            return f"({_expr(a)} * {_expr(b)})"
        case App(App(Prim("sub"), a), b):
            # monus — C has no native monus
            return f"({_expr(a)} > {_expr(b)} ? {_expr(a)} - {_expr(b)} : 0)"
        case App(App(Prim("eq_nat"), a), b):
            return f"({_expr(a)} == {_expr(b)})"
        case App(Prim("iszero"), a):
            return f"({_expr(a)} == 0)"
        case App(App(b, t1), t2):
            # Church-boolean application: if-then-else
            return f"({_expr(b)} ? {_expr(t1)} : {_expr(t2)})"
        case _:
            return f"/* {t!r} */ 0"


def emit_c(term, name: str = "f") -> str:
    """Emit a complete C translation unit for the synthesized term."""
    return (
        f"/* synthesized by arrow — shortest program consistent with examples */\n"
        f"int {name}(int x) {{\n"
        f"    return {_expr(term)};\n"
        f"}}\n"
    )

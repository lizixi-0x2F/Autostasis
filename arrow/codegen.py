"""Codegen — decompile a synthesized Term to readable C.

The Term is a pure expression over ints; the C output is a single
function. Nat semantics are preserved exactly: monus becomes saturating
subtraction, div/mod become zero-guarded, pow gets a helper function.
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
        case App(App(Prim("div"), a), b):
            # div by zero is 0 in Nat semantics; fold constant divisors
            if isinstance(b, Nat):
                return f"({_expr(a)} / {b.value})" if b.value != 0 else "0"
            return f"({_expr(b)} == 0 ? 0 : {_expr(a)} / {_expr(b)})"
        case App(App(Prim("mod"), a), b):
            if isinstance(b, Nat):
                return f"({_expr(a)} % {b.value})" if b.value != 0 else "0"
            return f"({_expr(b)} == 0 ? 0 : {_expr(a)} % {_expr(b)})"
        case App(App(Prim("pow"), a), b):
            return f"(ar_pow({_expr(a)}, {_expr(b)}))"
        case App(App(Prim("min"), a), b):
            return f"({_expr(a)} < {_expr(b)} ? {_expr(a)} : {_expr(b)})"
        case App(App(Prim("max"), a), b):
            return f"({_expr(a)} > {_expr(b)} ? {_expr(a)} : {_expr(b)})"
        case App(App(Prim("eq_nat"), a), b):
            return f"({_expr(a)} == {_expr(b)})"
        case App(App(Prim("le"), a), b):
            return f"({_expr(a)} <= {_expr(b)})"
        case App(App(Prim("lt"), a), b):
            return f"({_expr(a)} < {_expr(b)})"
        case App(App(Prim("ge"), a), b):
            return f"({_expr(a)} >= {_expr(b)})"
        case App(App(Prim("gt"), a), b):
            return f"({_expr(a)} > {_expr(b)})"
        case App(Prim("iszero"), a):
            return f"({_expr(a)} == 0)"
        case App(App(b, t1), t2):
            # Church-boolean application: if-then-else
            return f"({_expr(b)} ? {_expr(t1)} : {_expr(t2)})"
        case _:
            return f"/* {t!r} */ 0"


def _contains_pow(t) -> bool:
    match t:
        case App(f, a):
            if isinstance(f, Prim) and f.name == "pow":
                return True
            return _contains_pow(f) or _contains_pow(a)
        case _:
            return False


def emit_c(term, name: str = "f") -> str:
    """Emit a complete C translation unit for the synthesized term."""
    helper = ""
    if _contains_pow(term):
        helper = ("static int ar_pow(int b, int e) {\n"
                  "    int r = 1;\n"
                  "    while (e-- > 0) r *= b;\n"
                  "    return r;\n"
                  "}\n\n")
    return (
        f"/* synthesized by arrow — shortest program consistent with examples */\n"
        f"{helper}"
        f"int {name}(int x) {{\n"
        f"    return {_expr(term)};\n"
        f"}}\n"
    )

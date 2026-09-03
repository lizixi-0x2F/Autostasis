"""JSON and S-expression serialization for all term types."""
from __future__ import annotations
import json
from .terms import (Term, Var, Lam, App, Quote, Eval, Fix, Nat, Prim,
                     PartialPrim, Cons)


_TAG = "_tag"

def term_to_json(term: Term):
    match term:
        case Var(name):
            return {_TAG: "var", "name": name}
        case Lam(param, body):
            return {_TAG: "lam", "param": param, "body": term_to_json(body)}
        case App(func, arg):
            return {_TAG: "app", "func": term_to_json(func), "arg": term_to_json(arg)}
        case Quote(t):
            return {_TAG: "quote", "term": term_to_json(t)}
        case Eval(quoted, arg):
            return {_TAG: "eval", "quoted": term_to_json(quoted), "arg": term_to_json(arg)}
        case Fix(f):
            return {_TAG: "fix", "func": term_to_json(f)}
        case Nat(value):
            return {_TAG: "nat", "value": value}
        case Prim(name):
            return {_TAG: "prim", "name": name}
        case PartialPrim(name, arg1):
            return {_TAG: "partial_prim", "name": name, "arg1": term_to_json(arg1)}
        case Cons(car, cdr):
            return {_TAG: "cons", "car": term_to_json(car), "cdr": term_to_json(cdr)}
        case _:
            raise TypeError(f"Unknown term type: {type(term)}")


def term_from_json(data):
    tag = data[_TAG]
    match tag:
        case "var":    return Var(data["name"])
        case "lam":    return Lam(data["param"], term_from_json(data["body"]))
        case "app":    return App(term_from_json(data["func"]), term_from_json(data["arg"]))
        case "quote":  return Quote(term_from_json(data["term"]))
        case "eval":   return Eval(term_from_json(data["quoted"]), term_from_json(data["arg"]))
        case "fix":    return Fix(term_from_json(data["func"]))
        case "nat":    return Nat(data["value"])
        case "prim":   return Prim(data["name"])
        case "partial_prim": return PartialPrim(data["name"], term_from_json(data["arg1"]))
        case "cons":   return Cons(term_from_json(data["car"]), term_from_json(data["cdr"]))
        case _:
            raise ValueError(f"Unknown tag: {tag}")


def term_to_json_bytes(term: Term, indent: int | None = None) -> bytes:
    return json.dumps(term_to_json(term), ensure_ascii=False, indent=indent).encode("utf-8")


def term_from_json_bytes(data: bytes) -> Term:
    return term_from_json(json.loads(data.decode("utf-8")))


# ═══════════════════════════════════════════════════════════════════════════════
# S-expression output — human-readable, no parser needed

def term_to_sexpr(term: Term) -> str:
    match term:
        case Var(name):         return name
        case Lam(param, body):  return f"(lam {param} {term_to_sexpr(body)})"
        case App(func, arg):    return f"(app {term_to_sexpr(func)} {term_to_sexpr(arg)})"
        case Quote(t):          return f"(quote {term_to_sexpr(t)})"
        case Eval(q, a):        return f"(eval {term_to_sexpr(q)} {term_to_sexpr(a)})"
        case Fix(f):            return f"(fix {term_to_sexpr(f)})"
        case Nat(value):        return str(value)
        case Prim(name):        return f"#{name}"
        case PartialPrim(name, arg1): return f"(#{name} {term_to_sexpr(arg1)})"
        case Cons(car, cdr):    return f"(cons {term_to_sexpr(car)} {term_to_sexpr(cdr)})"
        case _:
            raise TypeError(f"Unknown term type: {type(term)}")

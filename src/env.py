"""Lexical environment with chained lookup."""
from __future__ import annotations
from typing import Dict
from .terms import Term


class Env:
    def __init__(self, bindings: Dict[str, Term] | None = None,
                 parent: Env | None = None):
        self.bindings: Dict[str, Term] = bindings or {}
        self.parent = parent

    def lookup(self, name: str) -> Term:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        raise ValueError(f"Unbound variable: {name}")

    def extend(self, name: str, value: Term) -> Env:
        return Env({name: value}, self)

    def extend_many(self, bindings: Dict[str, Term]) -> Env:
        return Env(bindings, self)


# ═══════════════════════════════════════════════════════════════════════════════
# Syntactic operations

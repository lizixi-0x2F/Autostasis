"""Fixed-point combinators and Church encodings."""
from __future__ import annotations
from .terms import Term, Var, Lam, App


Y_COMBINATOR: Term = Lam("f",
    App(
        Lam("x", App(Var("f"), App(Var("x"), Var("x")))),
        Lam("x", App(Var("f"), App(Var("x"), Var("x"))))
    )
)

Z_COMBINATOR: Term = Lam("f",
    App(
        Lam("x", App(Var("f"), Lam("v", App(App(Var("x"), Var("x")), Var("v"))))),
        Lam("x", App(Var("f"), Lam("v", App(App(Var("x"), Var("x")), Var("v")))))
    )
)

TRUE: Term = Lam("t", Lam("f", Var("t")))
FALSE: Term = Lam("t", Lam("f", Var("f")))


# ═══════════════════════════════════════════════════════════════════════════════

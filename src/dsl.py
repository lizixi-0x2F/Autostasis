"""Term construction DSL -- syntactic sugar for building Terms.

These are thin wrappers around Term constructors and Prims.
They live in src/ because they are the standard way to construct
arithmetic and function-space expressions.
"""
from .terms import App, Prim, Integ, Diff, SPACE_R, constant


# -- scalar constant --------------------------------------------------

def R(x):
    """Scalar constant on R."""
    return constant(x, space=SPACE_R)


# -- pointwise arithmetic ---------------------------------------------

def add(a, b): return App(App(Prim("addf"), a), b)
def sub(a, b): return App(App(Prim("subf"), a), b)
def mul(a, b): return App(App(Prim("mulf"), a), b)
def pow_(a, b): return App(App(Prim("powf"), a), b)


# -- function-space operators (first-class Terms, not Prims) ----------

def integ(f_lam, a, b=None):
    """Integral operator. integ(f, a) -> IVP, integ(f, a, b) -> BVP."""
    return Integ(f_lam, a, b)


def diff_op(f_lam):
    """Derivative operator."""
    return Diff(f_lam)

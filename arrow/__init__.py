"""Arrow — a synthesis compiler: examples in, C out.

    python -m arrow.main spec.arr -o out.c

Spec format:

    int f(int x)          # optional signature line
    f(0) = 1              # example lines: f(<input>) = <output>
    f(1) = 3
    # comments start with '#'

Arrow synthesizes the SHORTEST term (MDL induction, Occam's razor)
consistent with every example, then decompiles it to readable C.
Compiled with the same three-phase shape as a real compiler:
frontend (parse) -> synthesis -> codegen.
"""

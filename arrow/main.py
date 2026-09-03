"""Arrow compiler driver.

    python -m arrow.main spec.arr -o out.c

Arrow's core behavior is the predict-correct loop: examples are revealed
one at a time, the hypothesis predicts the next output, and mismatches
correct it by resynthesis. The final C is written to stdout or -o FILE.

Exit codes (compiler convention): 0 success, 1 synthesis failed, 2 parse error.
Diagnostics go to stderr, generated C to stdout or -o file.
"""
from __future__ import annotations
import argparse
import sys

from .frontend import parse_spec, ParseError
from .synth import SynthesisError
from .predict import predict_loop, report
from .codegen import emit_c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="arrow",
        description="synthesis compiler: examples in, C out (MDL induction)")
    ap.add_argument("spec", help="arrow spec file (examples)")
    ap.add_argument("-o", "--output", metavar="FILE",
                    help="write C to FILE (default: stdout)")
    ap.add_argument("--max-size", type=int, default=12, metavar="N",
                    help="maximum synthesized program size (default 12)")
    args = ap.parse_args(argv)

    # ── phase 1: frontend ──
    try:
        with open(args.spec, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"arrow: error: cannot open {args.spec!r}: {e.strerror}",
              file=sys.stderr)
        return 2

    try:
        name, examples = parse_spec(text, args.spec)
    except ParseError as e:
        print(e, file=sys.stderr)
        return 2

    # ── phase 2: predict-correct loop ──
    try:
        term, trace, hits, corrections = predict_loop(examples,
                                                      max_size=args.max_size)
    except SynthesisError as e:
        print(f"arrow: error: {e}", file=sys.stderr)
        return 1

    print(report(trace, name, hits, corrections), file=sys.stderr)

    # ── phase 3: codegen ──
    c_code = emit_c(term, name)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(c_code)
        except OSError as e:
            print(f"arrow: error: cannot write {args.output!r}: {e.strerror}",
                  file=sys.stderr)
            return 2
    else:
        sys.stdout.write(c_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())

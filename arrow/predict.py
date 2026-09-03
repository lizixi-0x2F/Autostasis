"""Predict-correct loop — Arrow's core behavior.

    examples are revealed one at a time. Arrow keeps a single hypothesis
    (the shortest program consistent with everything seen so far),
    predicts the next output, and on mismatch corrects itself by
    resynthesis. Self-reference lives here: the program's next shape is
    causally determined by its own previous prediction failures.

    p_{n+1} = shortest program consistent with all data revealed
              through p_n's failures
"""
from __future__ import annotations
from .synth import synthesize, run_nat, size, SynthesisError


def predict_loop(examples: list[tuple[int, int]], max_size: int = 12,
                 verbose: bool = True):
    """Reveal examples one by one; predict; correct on mismatch.

    Returns (program, trace, hits, corrections) where trace entries are
    ("init" | "hit" | "correct", x, y, y_hat, program_after, size_after).
    """
    if not examples:
        raise SynthesisError("no examples")
    seen = [examples[0]]
    prog, _, _ = synthesize(seen, max_size=max_size)
    hits = corrections = 0
    trace = [("init", examples[0][0], examples[0][1], None, prog, size(prog))]

    for x, y in examples[1:]:
        y_hat = run_nat(prog, x)
        seen.append((x, y))
        if y_hat == y:
            hits += 1
            trace.append(("hit", x, y, y_hat, prog, size(prog)))
        else:
            corrections += 1
            prog, _, _ = synthesize(seen, max_size=max_size)
            trace.append(("correct", x, y, y_hat, prog, size(prog)))

    return prog, trace, hits, corrections


def report(trace, name: str, hits: int, corrections: int) -> str:
    """Human-readable history of the program's self-corrections."""
    lines = []
    for ev in trace:
        kind, x, y, y_hat, prog, sz = ev
        if kind == "init":
            lines.append(f"arrow: saw {name}({x}) = {y}; hypothesis: {prog} (size {sz})")
        elif kind == "hit":
            lines.append(f"arrow: predict {name}({x}) = {y_hat} ... actual {y} — hit")
        else:
            lines.append(f"arrow: predict {name}({x}) = {y_hat} ... actual {y} — MISMATCH")
            lines.append(f"arrow: corrected hypothesis: {prog} (size {sz})")
    lines.append(f"arrow: final program (size {trace[-1][5]}, "
                 f"{hits} hit(s), {corrections} correction(s))")
    return "\n".join(lines)

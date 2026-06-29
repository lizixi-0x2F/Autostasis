"""Reverse-mode automatic differentiation tape.

Tape: Wengert list — records scalar ops during forward pass, replays backward
      in one reverse sweep. O(1) gradient for any number of parameters.

TapeVar: A scalar value on the tape. Overloads arithmetic (+, -, *, /, **)
         and math functions (sin, cos, exp, tanh, relu...) so that the
         computation graph is built automatically during forward evaluation.

Usage:
    tape = Tape()
    θ = TapeVar(0.5, tape, tape.var(0.5, "θ"))  # parameter
    x = TapeVar.const(3.0)                        # constant (no gradient)
    y = θ * x + θ.sin()                           # records mul, sin, add
    tape.backward()                               # reverse sweep
    print(tape.gradient(θ.tape_id))               # ∂y/∂θ
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Tape
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TapeEntry:
    op: str
    inputs: tuple   # tape indices of inputs
    value: float
    meta: dict = field(default_factory=dict)


class Tape:
    """Reverse-mode computation tape (Wengert list).

    Forward: record operations via TapeVar arithmetic.
    Backward: replay in reverse, accumulating gradients via chain rule.
    """
    def __init__(self):
        self.entries: list[TapeEntry] = []
        self._grad: dict[int, float] = {}

    def var(self, value: float, name: str = "") -> int:
        """Create a leaf variable. Its gradient will be computed."""
        idx = len(self.entries)
        self.entries.append(TapeEntry("var", (), value, {"name": name}))
        return idx

    def record(self, op: str, value: float, *inputs: int, **meta) -> int:
        """Record an operation. Returns tape index for the result.
        Usage: tape.record("add", result, left_id, right_id, a=..., b=...)
        """
        idx = len(self.entries)
        self.entries.append(TapeEntry(op, inputs, value, meta))
        return idx

    def backward(self):
        """Reverse through the tape, accumulating gradients.

        Seeds the last entry with gradient 1.0, then propagates
        backward through all recorded operations via the chain rule.
        """
        n = len(self.entries)
        if n == 0:
            return

        # Seed: ∂output/∂output = 1.0
        self._grad[n - 1] = 1.0

        for i in range(n - 1, -1, -1):
            dout = self._grad.pop(i, 0.0)
            if dout == 0.0:
                continue

            entry = self.entries[i]
            if entry.op == "var":
                self._grad[i] = self._grad.get(i, 0.0) + dout
                continue

            a = entry.inputs[0]
            b = entry.inputs[1] if len(entry.inputs) > 1 else -1

            def acc(j, v):
                if j >= 0:
                    self._grad[j] = self._grad.get(j, 0.0) + v

            # ── Backward rules for each operation ──
            m = entry.meta
            if entry.op == "add":
                acc(a, dout); acc(b, dout)
            elif entry.op == "sub":
                acc(a, dout); acc(b, -dout)
            elif entry.op == "mul":
                acc(a, dout * m["b"]); acc(b, dout * m["a"])
            elif entry.op == "div":
                acc(a, dout / m["b"]); acc(b, -dout * m["a"] / (m["b"] * m["b"]))
            elif entry.op == "pow":
                av, bv = m["a"], m["b"]
                # ∂/∂a a^b = b * a^(b-1)  (works for negative a if b is integer)
                acc(a, dout * bv * (av ** (bv - 1)) if av != 0 or bv > 1 else 0.0)
                # ∂/∂b a^b = a^b * log(a)  (only when a > 0)
                if av > 0:
                    acc(b, dout * entry.value * math.log(av))
            elif entry.op == "neg":
                acc(a, -dout)
            elif entry.op == "sin":
                acc(a, dout * m["cos_a"])
            elif entry.op == "cos":
                acc(a, -dout * m["sin_a"])
            elif entry.op == "exp":
                acc(a, dout * entry.value)
            elif entry.op == "log":
                acc(a, dout / m["a"] if m["a"] != 0 else 0.0)
            elif entry.op == "sqrt":
                acc(a, dout / (2.0 * entry.value) if entry.value != 0 else 0.0)
            elif entry.op == "tanh":
                acc(a, dout * (1.0 - entry.value ** 2))
            elif entry.op == "relu":
                acc(a, dout if m["a"] > 0 else 0.0)
            elif entry.op == "abs":
                acc(a, dout if m["a"] > 0 else -dout if m["a"] < 0 else 0.0)
            elif entry.op == "sum":
                for inp in entry.inputs:
                    acc(inp, dout)
            elif entry.op == "mean":
                n_in = len(entry.inputs)
                for inp in entry.inputs:
                    acc(inp, dout / n_in)

    def gradient(self, var_idx: int) -> float:
        """Get accumulated gradient for a tape variable."""
        return self._grad.get(var_idx, 0.0)

    def gradients(self, var_indices: list[int]) -> np.ndarray:
        """Get gradients for multiple tape variables as a numpy array."""
        return np.array([self.gradient(i) for i in var_indices])


# ═══════════════════════════════════════════════════════════════════════════════
# TapeVar — tracked scalar
# ═══════════════════════════════════════════════════════════════════════════════

class TapeVar:
    """A scalar that auto-records operations for reverse-mode AD.

    Overloaded operators (+, -, *, /, **) and math methods (sin, cos, exp,
    tanh, relu, etc.) all record entries on the attached Tape.

    Constants (created via TapeVar.const()) have tape=None and don't
    record operations. Mixed operations between tracked and constant
    vars are recorded on the tracked var's tape.
    """
    __slots__ = ("value", "tape", "tape_id")

    def __init__(self, value: float, tape: Tape | None, tape_id: int):
        self.value = float(value)
        self.tape = tape
        self.tape_id = tape_id

    @staticmethod
    def const(value: float) -> "TapeVar":
        """Create a constant — no gradient tracking."""
        return TapeVar(float(value), None, -1)

    def _has_tape(self) -> bool:
        return self.tape is not None and self.tape_id >= 0

    def _get_tape(self, other: "TapeVar") -> Tape | None:
        return self.tape if self.tape is not None else other.tape

    # ── Binary operations ──

    def _binop(self, other, op: str, fn, **meta):
        if isinstance(other, (int, float, np.floating)):
            other = TapeVar.const(float(other))
        if not isinstance(other, TapeVar):
            return NotImplemented
        result_val = fn(self.value, other.value)
        if not self._has_tape() and not other._has_tape():
            return TapeVar.const(result_val)
        tape = self._get_tape(other)
        tid = tape.record(op, result_val, self.tape_id, other.tape_id,
                          a=self.value, b=other.value, **meta)
        return TapeVar(result_val, tape, tid)

    def __add__(self, other): return self._binop(other, "add", lambda a, b: a + b)
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return self._binop(other, "sub", lambda a, b: a - b)
    def __rsub__(self, other):
        if isinstance(other, (int, float)): return TapeVar.const(float(other)) - self
        return NotImplemented
    def __mul__(self, other): return self._binop(other, "mul", lambda a, b: a * b)
    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other): return self._binop(other, "div", lambda a, b: a / b if b != 0 else float("inf"))
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)): return TapeVar.const(float(other)) / self
        return NotImplemented
    def __neg__(self): return self._unop("neg", lambda a: -a)
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return self._binop(TapeVar.const(float(other)), "pow",
                              lambda a, b: a ** b if (a >= 0 or float(int(b)) == b) else float("nan"))
        return self._binop(other, "pow", lambda a, b: a ** b if (a >= 0 or float(int(b)) == b) else float("nan"))

    # ── Unary operations ──

    def _unop(self, op: str, fn, **meta):
        result_val = fn(self.value)
        if not self._has_tape():
            return TapeVar.const(result_val)
        tid = self.tape.record(op, result_val, self.tape_id, **meta)
        return TapeVar(result_val, self.tape, tid)

    def sin(self):   return self._unop("sin", math.sin, cos_a=math.cos(self.value))
    def cos(self):   return self._unop("cos", math.cos, sin_a=math.sin(self.value))
    def exp(self):   return self._unop("exp", math.exp)
    def log(self):   return self._unop("log", math.log, a=self.value)
    def sqrt(self):  return self._unop("sqrt", math.sqrt)
    def tanh(self):  return self._unop("tanh", math.tanh)
    def relu(self):  return self._unop("relu", lambda a: max(0.0, a), a=self.value)

    def __repr__(self):
        return f"TapeVar({self.value:.6f})"


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorized operations on lists of TapeVars
# ═══════════════════════════════════════════════════════════════════════════════

def tv_dot(xs: list[TapeVar], ys: list[TapeVar]) -> TapeVar:
    """Dot product of two lists of TapeVars. Records sum of element-wise muls."""
    assert len(xs) == len(ys)
    result = xs[0] * ys[0]
    for i in range(1, len(xs)):
        result = result + xs[i] * ys[i]
    return result


def tv_sum(xs: list[TapeVar]) -> TapeVar:
    """Sum a list of TapeVars. Records individual adds."""
    result = xs[0]
    for i in range(1, len(xs)):
        result = result + xs[i]
    return result


def tv_mean(xs: list[TapeVar]) -> TapeVar:
    """Mean of a list of TapeVars."""
    return tv_sum(xs) / len(xs)


def tv_mse(preds: list[TapeVar], targets: list[float]) -> TapeVar:
    """Mean squared error. preds: list of TapeVars, targets: list of floats."""
    total = (preds[0] - targets[0]) ** 2
    for i in range(1, len(preds)):
        total = total + (preds[i] - targets[i]) ** 2
    return total / len(preds)


def tv_bce(logits: list[TapeVar], targets: list[float]) -> TapeVar:
    """Binary cross-entropy with logits.

    BCE(z, y) = -(y·log(σ(z)) + (1-y)·log(1-σ(z)))
    where σ(z) = 1/(1+exp(-z)).

    All ops are TapeVar ops → auto-recorded on tape.
    """
    n = len(logits)
    total = _bce_one(logits[0], targets[0])
    for i in range(1, n):
        total = total + _bce_one(logits[i], targets[i])
    return total / n


def _bce_one(z: TapeVar, y: float) -> TapeVar:
    """BCE for one sample: σ(z) = 1/(1+e^{-z}), BCE = -(y·log(σ) + (1-y)·log(1-σ))."""
    one = TapeVar.const(1.0)
    sig = one / (one + (-z).exp())
    y_tv = TapeVar.const(float(y))
    return -(y_tv * sig.log() + (one - y_tv) * (one - sig).log())


def tv_softmax_ce(logits: list[TapeVar], target: int) -> TapeVar:
    """Softmax cross-entropy for one sample. O(1) gradient via tape.

    logits: list of TapeVar, length = vocab_size
    target: int, the correct class index

    Computes: -log(softmax(logits)[target])
    Numerically stable via log-softmax trick.
    """
    # max(logits) for numerical stability
    max_val = max(l.value for l in logits)
    shifted = [l - max_val for l in logits]
    exps = [s.exp() for s in shifted]
    sum_exp = exps[0]
    for e in exps[1:]:
        sum_exp = sum_exp + e
    log_sum = sum_exp.log()
    return -(shifted[target] - log_sum)


def tv_softmax_ce_batch(logits_batch: list[list[TapeVar]],
                        targets: list[int]) -> TapeVar:
    """Average softmax CE over a batch."""
    n = len(logits_batch)
    total = tv_softmax_ce(logits_batch[0], targets[0])
    for i in range(1, n):
        total = total + tv_softmax_ce(logits_batch[i], targets[i])
    return total / n

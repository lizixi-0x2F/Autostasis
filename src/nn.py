"""Models — define in numpy, train via Fix(T).

Model: abstract base. forward(params, x) takes np.ndarray, returns float.
FunctionModel: wrap any numpy function as a Model. Gradient via finite differences.

Usage:
    # One-liner
    model = FunctionModel(lambda p, x: p[0]*x + p[1], n_params=2)
    trainer = Trainer(model, lr=0.1)
    trainer.fit(X, Y)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """Abstract model. forward takes flat np.ndarray params, returns float."""

    @abstractmethod
    def param_init(self, rng: np.random.RandomState = None) -> np.ndarray:
        """Flat parameter vector."""
        ...

    @abstractmethod
    def n_params(self) -> int:
        """Number of scalar parameters."""
        ...

    @abstractmethod
    def forward(self, params: np.ndarray, x: float) -> float:
        """Forward pass: params (p,) + x → scalar output."""
        ...

    def grad(self, params: np.ndarray, X: np.ndarray, Y: np.ndarray,
             loss: str = "mse", h: float = 1e-5) -> np.ndarray:
        """Gradient of loss w.r.t. params via central finite differences.

        Default implementation works for any forward() — no tape needed.
        O(p) per call. Override for efficiency (e.g., tape-based O(1)).
        """
        p = self.n_params()
        base = self._loss(params, X, Y, loss)
        grad = np.zeros(p)
        for i in range(p):
            params_plus = params.copy()
            params_plus[i] += h
            loss_plus = self._loss(params_plus, X, Y, loss)
            grad[i] = (loss_plus - base) / h
        return grad

    def _loss(self, params: np.ndarray, X: np.ndarray, Y: np.ndarray,
              loss: str = "mse") -> float:
        preds = np.array([self.forward(params, float(x)) for x in X])
        Y = np.asarray(Y, dtype=float).ravel()
        if loss == "mse":
            return float(np.mean((preds - Y) ** 2))
        elif loss == "ce":
            return self._bce(preds, Y)
        raise ValueError(f"Unknown loss: {loss}")

    @staticmethod
    def _bce(logits: np.ndarray, Y: np.ndarray) -> float:
        """Binary cross-entropy with logits (numerically stable)."""
        # BCE(logit, y) = max(z,0) - z*y + log(1+exp(-|z|))
        z = np.asarray(logits, dtype=float).ravel()
        y = np.asarray(Y, dtype=float).ravel()
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


# ═══════════════════════════════════════════════════════════════════════════════
# FunctionModel — wrap any numpy function as a Model
# ═══════════════════════════════════════════════════════════════════════════════

class FunctionModel(Model):
    """Wrap a numpy (params, x) -> float function as a Model.

    Gradient via finite differences — works for ANY function, no tape needed.

    Example:
        # Linear
        model = FunctionModel(lambda p, x: p[0]*x + p[1], n_params=2)

        # MLP (pure numpy)
        def mlp_fn(p, x):
            W1 = p[:16].reshape(1, 16); b1 = p[16:32]
            W2 = p[32:48].reshape(16, 1); b2 = p[48]
            h = np.tanh(x * W1 + b1)
            return float(h @ W2 + b2)
        model = FunctionModel(mlp_fn, n_params=49)

        # DEQ (pure numpy)
        def deq_fn(p, x, max_iter=50, tol=1e-6):
            w, u, b = p[0], p[1], p[2]
            z = 0.0
            for _ in range(max_iter):
                z_new = np.tanh(w*z + u*x + b)
                if abs(z_new - z) < tol: return float(z_new)
                z = z_new
            return float(z)
        model = FunctionModel(deq_fn, n_params=3)
    """

    def __init__(self, forward_fn,
                 n_params: int,
                 param_init: callable = None):
        self._forward_fn = forward_fn
        self._n_params = n_params
        self._param_init = param_init

    def n_params(self) -> int:
        return self._n_params

    def param_init(self, rng: np.random.RandomState = None) -> np.ndarray:
        if self._param_init is not None:
            if rng is None:
                rng = np.random.RandomState(42)
            return np.asarray(self._param_init(rng), dtype=float)
        if rng is None:
            rng = np.random.RandomState(42)
        return rng.uniform(-0.5, 0.5, self._n_params)

    def forward(self, params: np.ndarray, x: float) -> float:
        return float(self._forward_fn(np.asarray(params, dtype=float), x))


# ═══════════════════════════════════════════════════════════════════════════════
# TapeModel — tape-based O(1) gradient (for TapeVar-compatible functions)
# ═══════════════════════════════════════════════════════════════════════════════

class TapeModel(Model):
    """Model with tape-based O(1) gradient via reverse-mode AD.

    The forward function uses TapeVar operations (which look like numpy
    but auto-record on a tape). Gradient computed in one reverse sweep.

    Example:
        from autostasis.grad import TapeVar
        model = TapeModel(lambda p, x: p[0]*x + p[1], n_params=2)
    """

    def __init__(self, forward_fn,
                 n_params: int,
                 param_init: callable = None):
        self._forward_fn = forward_fn
        self._n_params = n_params
        self._param_init = param_init

    def n_params(self) -> int:
        return self._n_params

    def param_init(self, rng: np.random.RandomState = None) -> np.ndarray:
        if self._param_init is not None:
            if rng is None:
                rng = np.random.RandomState(42)
            return np.asarray(self._param_init(rng), dtype=float)
        if rng is None:
            rng = np.random.RandomState(42)
        return rng.uniform(-0.5, 0.5, self._n_params)

    def forward(self, params: np.ndarray, x: float) -> float:
        # For prediction (no tape): just call the function
        from .grad import TapeVar
        p_tv = [TapeVar(float(params[i]), None, -1) for i in range(self._n_params)]
        result = self._forward_fn(p_tv, x)
        return float(result.value)

    def grad(self, params: np.ndarray, X: np.ndarray, Y: np.ndarray,
             loss: str = "mse", **kw) -> np.ndarray:
        """O(1) gradient via tape.backward()."""
        from .grad import Tape, TapeVar, tv_mse, tv_bce

        p = self._n_params
        tape = Tape()
        params_tv = [TapeVar(float(params[i]), tape, tape.var(float(params[i])))
                     for i in range(p)]

        preds = [self._forward_fn(params_tv, float(xi)) for xi in X]
        Y_list = [float(y) for y in Y]
        if loss == "ce":
            loss_val = tv_bce(preds, Y_list)
        else:
            loss_val = tv_mse(preds, Y_list)
        tape.backward()

        return np.array([tape.gradient(params_tv[i].tape_id) for i in range(p)])

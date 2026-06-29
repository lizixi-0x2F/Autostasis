"""Trainer — one class, any model.

  Forward:  model.forward(params, x) → float
  Gradient: model.grad(params, X, Y) → np.ndarray
  Update:   θ ← θ − η·∇L  (T-transform)

Usage:
    model = FunctionModel(lambda p, x: p[0]*x + p[1], n_params=2)
    trainer = Trainer(model, lr=0.1)
    params = trainer.fit(X, Y, epochs=50)
"""
import numpy as np
from .nn import Model


class Trainer:
    """Trainer: model.fit(X, Y) — dead simple."""

    def __init__(self, model: Model, lr: float = 0.01, loss: str = "mse"):
        """
        Args:
            model: Model instance
            lr: learning rate
            loss: "mse" (regression) or "ce" (binary cross-entropy with logits)
        """
        self.model = model
        self.lr = lr
        self.loss_name = loss
        if loss not in ("mse", "ce"):
            raise ValueError(f"Unknown loss: {loss}. Use 'mse' or 'ce'.")
        self._params: np.ndarray | None = None
        self._loss_history: list[float] = []

    @property
    def params(self) -> np.ndarray:
        if self._params is None:
            self._params = self.model.param_init()
        return self._params

    @params.setter
    def params(self, value: np.ndarray):
        self._params = np.asarray(value, dtype=float)

    def fit(self, X: np.ndarray, Y: np.ndarray,
            epochs: int = 100, verbose: bool = True,
            tol: float = 1e-8) -> np.ndarray:
        """Train on supervised data (X, Y).

        Each epoch: grad = model.grad(params, X, Y), then θ ← θ − η·grad.
        """
        X = np.asarray(X, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()

        if self._params is None:
            self._params = self.model.param_init()

        self._loss_history = []

        if verbose:
            print(f"Trainer.fit: {self.model.__class__.__name__}")
            print(f"  n_params={self.model.n_params()}, lr={self.lr}")
            print(f"  n_samples={len(X)}, epochs={epochs}")
            print(f"  {'epoch':>6s}  {'loss':>14s}  {'|grad|':>14s}  {'|delta|':>14s}")

        for epoch in range(epochs):
            grads = self.model.grad(self._params, X, Y, self.loss_name)
            loss_val = self.model._loss(self._params, X, Y, self.loss_name)
            grad_norm = float(np.linalg.norm(grads))
            self._loss_history.append(loss_val)

            # T-transform
            delta = self.lr * grads
            self._params = self._params - delta
            delta_norm = float(np.linalg.norm(delta))

            if verbose and (epoch < 10 or epoch % max(1, epochs // 10) == 0
                           or grad_norm < tol * 100):
                print(f"  {epoch:>6d}  {loss_val:>14.8f}  "
                      f"{grad_norm:>14.6e}  {delta_norm:>14.6e}")

            if grad_norm < tol:
                if verbose:
                    print(f"  ✓ Converged at epoch {epoch + 1}")
                break

        if verbose:
            print(f"  Final loss: {self._loss_history[-1]:.8f}")

        return self._params

    def fit_steps(self, n_steps: int = 100, verbose: bool = True,
                  tol: float = 1e-8) -> np.ndarray:
        """Train step-by-step. Model handles data sampling internally.

        Each step: grad = model.grad(params), then θ ← θ − η·grad.
        No X,Y passed — the model owns its data (e.g., GPT samples sequences).
        """
        if self._params is None:
            self._params = self.model.param_init()

        self._loss_history = []

        if verbose:
            print(f"Trainer.fit_steps: {self.model.__class__.__name__}")
            print(f"  n_params={self.model.n_params()}, lr={self.lr}")
            print(f"  n_steps={n_steps}")
            print(f"  {'step':>6s}  {'loss':>14s}  {'|grad|':>14s}  {'|delta|':>14s}")

        for step in range(n_steps):
            grads = self.model.grad(self._params)
            loss_val = self.model._loss(self._params)
            grad_norm = float(np.linalg.norm(grads))
            self._loss_history.append(loss_val)

            # T-transform
            delta = self.lr * grads
            self._params = self._params - delta
            delta_norm = float(np.linalg.norm(delta))

            if verbose and (step < 10 or step % max(1, n_steps // 10) == 0
                           or grad_norm < tol * 100):
                print(f"  {step:>6d}  {loss_val:>14.6f}  "
                      f"{grad_norm:>14.4e}  {delta_norm:>14.4e}")

            if grad_norm < tol:
                if verbose:
                    print(f"  Converged at step {step + 1}")
                break

        if verbose:
            print(f"  Final loss: {self._loss_history[-1]:.6f}")

        return self._params

    def predict(self, params: np.ndarray = None, X: np.ndarray = None
                ) -> np.ndarray:
        if params is None:
            params = self.params
        if X is None:
            raise ValueError("X required")
        X = np.asarray(X, dtype=float).ravel()
        return np.array([self.model.forward(params, float(x)) for x in X])

    @property
    def loss_history(self) -> list[float]:
        return self._loss_history

"""Trainer — one class, any model.

  Learning is a fixed point:

      Forward:  model.forward(params, x) → float
      Gradient: model.grad(params, X, Y) → np.ndarray
      Solution: θ* = Fix(T),   T[θ] = θ − η·A[θ]   (A = ∇_θ L, a numerical Fun)

  The convergence engine is eval.fixpoint — the same engine that solves every
  equation A[u] = 0. The Trainer owns no iteration loop; it owns the
  construction of the term Fix(T).

Usage:
    model = FunctionModel(lambda p, x: p[0]*x + p[1], n_params=2)
    trainer = Trainer(model, lr=0.1)
    params = trainer.fit(X, Y, epochs=50)
"""
import numpy as np
from .nn import Model
from .terms import Var, Lam, App, Fix, Fun, SPACE_RN
from .dsl import R, sub, mul
from .eval import fixpoint


class Trainer:
    """Trainer: model.fit(X, Y) — learning is Fix(T)."""

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

    # ── The whole of learning: one term, one fixed point ──

    def _fixed_point(self, grad_op, *, tol, max_iter, verbose,
                     on_iter=None, label="Fix(T)"):
        """θ* = Fix(T), T[θ] = θ − η·A[θ].

        A is the gradient operator, a numerical Fun on SPACE_RN(n). Function
        composition App(A, θ) is exactly "feed θ into the gradient" — θ is a
        constant vector Fun, so A∘θ(x) = ∇L(θ) for every x.
        """
        n = self.model.n_params()
        sp = SPACE_RN(n)
        th = Var("th")
        A = Fun(lambda v, _g=grad_op: np.asarray(_g(v), dtype=float), space=sp)
        T = Lam("th", sub(th, mul(R(self.lr), App(A, th))))
        theta0 = Fun(lambda _, _v=self._params.copy(): _v, space=sp)
        if verbose:
            print(f"{label}")
            print(f"  n_params={n}, lr={self.lr}")
            print(f"  {'iter':>6s}  {'loss':>14s}  {'|delta|':>14s}")
        return fixpoint(Fix(T), theta0, tol=tol, max_iter=max_iter,
                        on_iter=on_iter)

    def fit(self, X: np.ndarray, Y: np.ndarray,
            epochs: int = 100, verbose: bool = True,
            tol: float = 1e-8) -> np.ndarray:
        """Train on supervised data (X, Y).

        The update T[θ] = θ − η·∇L is built as a Term; convergence is driven
        by eval.fixpoint — no loop lives in this method.
        """
        X = np.asarray(X, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()

        if self._params is None:
            self._params = self.model.param_init()

        self._loss_history = []
        loss_name = self.loss_name

        def grad_op(th_vec):
            return self.model.grad(np.asarray(th_vec, dtype=float),
                                   X, Y, loss_name)

        def on_iter(i, x_old, x_new, change):
            loss_val = self.model._loss(np.asarray(x_new(0), dtype=float),
                                        X, Y, loss_name)
            self._loss_history.append(loss_val)
            if verbose and (i < 10 or i % max(1, epochs // 10) == 0
                            or change < tol * 100):
                print(f"  {i:>6d}  {loss_val:>14.8f}  {change:>14.6e}")

        if verbose:
            print(f"Trainer.fit: {self.model.__class__.__name__}")
            print(f"  n_samples={len(X)}, epochs={epochs}")

        theta_star = self._fixed_point(grad_op, tol=tol, max_iter=epochs,
                                       verbose=verbose, on_iter=on_iter)
        if theta_star is None:
            if verbose:
                print("  X diverged")
            return self._params
        self._params = np.asarray(theta_star(0), dtype=float)
        if verbose:
            print(f"  Final loss: {self._loss_history[-1]:.8f}")
        return self._params

    def fit_steps(self, n_steps: int = 100, verbose: bool = True,
                  tol: float = 1e-8) -> np.ndarray:
        """Train step-by-step; the model owns its data (e.g., GPT samples).

        Same Fix(T) engine — the model supplies its own gradients.
        """
        if self._params is None:
            self._params = self.model.param_init()

        self._loss_history = []

        def grad_op(th_vec):
            return self.model.grad(np.asarray(th_vec, dtype=float))

        def on_iter(i, x_old, x_new, change):
            loss_val = self.model._loss(np.asarray(x_new(0), dtype=float))
            self._loss_history.append(loss_val)
            if verbose and (i < 10 or i % max(1, n_steps // 10) == 0
                            or change < tol * 100):
                print(f"  {i:>6d}  {loss_val:>14.6f}  {change:>14.6e}")

        if verbose:
            print(f"Trainer.fit_steps: {self.model.__class__.__name__}")
            print(f"  n_steps={n_steps}")

        theta_star = self._fixed_point(grad_op, tol=tol, max_iter=n_steps,
                                       verbose=verbose, on_iter=on_iter)
        if theta_star is None:
            if verbose:
                print("  X diverged")
            return self._params
        self._params = np.asarray(theta_star(0), dtype=float)
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

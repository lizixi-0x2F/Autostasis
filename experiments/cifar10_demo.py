"""CIFAR-10 classifier via TapeModel + Trainer.fit().

Downsampled 16x16 grayscale -> MLP[256, 64, 10] -> softmax CE.
All ops = TapeVar ops -> one tape.backward() -> O(1) gradient.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.nn import TapeModel
from src.trainer import Trainer
from src.grad import TapeVar, Tape, tv_softmax_ce


# ---- Load CIFAR-10 ----
def load_cifar10(n_train=2000, n_test=400):
    """Load CIFAR-10, grayscale, 16x16 downsample, flatten to 256."""
    from torchvision import datasets
    train_set = datasets.CIFAR10(root="/tmp/cifar10", train=True, download=True)
    test_set = datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True)

    def preprocess(data):
        X = data.astype(np.float32).mean(axis=-1) / 255.0  # grayscale
        X = X.reshape(-1, 16, 2, 16, 2).mean(axis=(2, 4))   # 32x32 -> 16x16
        return X.reshape(-1, 256)

    Xtr = preprocess(train_set.data[:n_train])
    Ytr = np.array(train_set.targets[:n_train])
    Xte = preprocess(test_set.data[:n_test])
    Yte = np.array(test_set.targets[:n_test])
    return Xtr, Ytr, Xte, Yte

X_train, Y_train, X_test, Y_test = load_cifar10()
INPUT_DIM = 256
N_CLASSES = 10
N_PARAMS = INPUT_DIM * 64 + 64 + 64 * N_CLASSES + N_CLASSES  # 256*64+64+64*10+10 = 17162

print(f"CIFAR-10: {len(X_train)} train, {len(X_test)} test")
print(f"Input: {INPUT_DIM} (16x16 grayscale), Classes: {N_CLASSES}")
print(f"MLP: [{INPUT_DIM}, 64, {N_CLASSES}], params={N_PARAMS}")


# ---- MLP forward with TapeVars ----
def mlp_forward(p, x_flat):
    """p: list[TapeVar], x_flat: list[float]. Returns list[TapeVar] (10 logits)."""
    # Layer 1: W1 (256,64), b1 (64)
    W1 = [[p[r * 64 + c] for c in range(64)] for r in range(256)]
    b1 = [p[16384 + c] for c in range(64)]  # 256*64 = 16384
    # Layer 2: W2 (64,10), b2 (10)
    off2 = 16384 + 64
    W2 = [[p[off2 + r * 10 + c] for c in range(10)] for r in range(64)]
    b2 = [p[off2 + 640 + c] for c in range(10)]  # 64*10 = 640

    # Hidden: h = relu(x @ W1 + b1)
    hidden = []
    for j in range(64):
        z = b1[j]
        for i in range(256):
            z = z + x_flat[i] * W1[i][j]
        hidden.append(z.relu())

    # Output: logits = h @ W2 + b2
    logits = []
    for j in range(10):
        z = b2[j]
        for i in range(64):
            z = z + hidden[i] * W2[i][j]
        logits.append(z)

    return logits


# ---- Training ----

# Convert X_train to list of list[float] for TapeVar const
X_train_list = [[float(v) for v in row] for row in X_train[:1000]]
Y_train_list = [int(y) for y in Y_train[:1000]]

class MLPModel(TapeModel):
    def __init__(self):
        super().__init__(mlp_forward, n_params=N_PARAMS)
        self.X = X_train_list
        self.Y = Y_train_list

    def _softmax_ce_numpy(self, logits, target):
        """Numerically stable softmax cross-entropy in pure numpy."""
        x = np.array(logits, dtype=float)
        x_max = x.max()
        shifted = x - x_max
        exp_shifted = np.exp(shifted)
        softmax = exp_shifted / exp_shifted.sum()
        return -np.log(max(softmax[target], 1e-15))

    def forward(self, params, x=None):
        """Return softmax CE loss for a single sample (no tape)."""
        idx = np.random.randint(0, len(self.X)) if x is None else int(x) % len(self.X)
        p_tv = [TapeVar(float(params[i]), None, -1) for i in range(N_PARAMS)]
        logits_tv = self._forward_fn(p_tv, self.X[idx])
        logits = [lt.value for lt in logits_tv]
        return float(self._softmax_ce_numpy(logits, self.Y[idx]))

    def _loss(self, params, X=None, Y=None, loss="ce"):
        n = min(200, len(self.X))
        return sum(self.forward(params, i) for i in range(n)) / n

    def grad(self, params, X=None, Y=None, loss="ce", **kw):
        """Tape O(1) gradient on one random sample."""
        idx = np.random.randint(0, len(self.X))
        tape = Tape()
        p_tv = [TapeVar(float(params[i]), tape, tape.var(float(params[i])))
                for i in range(N_PARAMS)]
        logits = mlp_forward(p_tv, [TapeVar.const(v) for v in self.X[idx]])
        loss_tv = tv_softmax_ce(logits, self.Y[idx])
        tape.backward()
        return np.array([tape.gradient(p_tv[i].tape_id) for i in range(N_PARAMS)])


model = MLPModel()
trainer = Trainer(model, lr=0.1, loss="ce")
params = trainer.fit_steps(n_steps=300)

# ---- Evaluate ----
correct = 0
for i in range(len(X_test)):
    p_tv = [TapeVar(float(params[j]), None, -1) for j in range(N_PARAMS)]
    logits = mlp_forward(p_tv, [float(v) for v in X_test[i]])
    pred = max(range(10), key=lambda j: logits[j].value)
    if pred == Y_test[i]: correct += 1

acc = correct / len(X_test)
print(f"\nTest accuracy: {acc:.2%} ({correct}/{len(X_test)})")

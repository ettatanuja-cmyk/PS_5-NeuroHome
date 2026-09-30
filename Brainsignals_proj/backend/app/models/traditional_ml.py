"""Traditional ML baselines (base paper): Logistic Regression, SVM, MLP.

Pure-NumPy implementations (no sklearn/scipy wheels needed — those are
blocked by Windows App Control on some machines):
  * ``logreg``: multinomial softmax (logistic) regression, Adam.
  * ``svm``: RBF-kernel SVM via Random Fourier Features (Rahimi-Recht
    approximation) + one-vs-rest hinge loss, Adam.
  * ``mlp``: 2-hidden-layer perceptron (d->128->64->5, ReLU), Adam.

Each classifier works on a fixed window-feature vector extracted from the
preprocessed EEG (band powers, amplitude statistics, Hjorth parameters,
zero-crossings, spectral edge), standardised with a saved scaler. At
inference, probabilities are averaged over evenly-spaced windows covering
the whole file. Checkpoints: ``weights/traditional_ml.pkl`` (all three
models + scaler + feature names).
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import numpy as np

from .common import (
    TARGET_CLASSES,
    WEIGHTS_DIR,
    _fit_channels,
    _format_prediction,
    _numpy_fallback_logits,
    _softmax,
    _windows,
)

FEATURE_NAMES: List[str] = []
for _band in ("broadband", "delta", "theta", "alpha", "beta"):
    for _stat in ("mean", "std", "ptp", "zcr", "hjorth_mob"):
        FEATURE_NAMES.append(f"chmean_{_band}_{_stat}")
FEATURE_NAMES += ["n_channels", "spectral_edge", "line_length", "kurtosis"]
N_FEATURES = len(FEATURE_NAMES)

BANDS = {
    "broadband": (1.0, 15.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 15.0),
}
FS = 120.0
CKPT_NAME = "traditional_ml.pkl"


def _bandpower_fft(x: np.ndarray, fs: float, low: float,
                   high: float) -> np.ndarray:
    """Mean band power per channel via rFFT (no scipy needed)."""
    n = x.shape[1]
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    psd = np.abs(np.fft.rfft(x, axis=1)) ** 2 / max(n, 1)
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return np.zeros(x.shape[0])
    return psd[:, mask].mean(axis=1)


def extract_window_features(win: np.ndarray) -> np.ndarray:
    """Fixed feature vector for one (C, T) window. Pure NumPy."""
    x = _fit_channels(np.asarray(win, dtype=np.float64))
    feats: List[float] = []
    for low, high in BANDS.values():
        _ = _bandpower_fft(x, FS, low, high)  # layout slot per band
        feats.append(float(np.mean(x)))
        feats.append(float(np.mean((x - x.mean()) ** 2)))
        feats.append(float(np.mean(np.ptp(x, axis=1))))
        zc = ((x[:, :-1] > 0) != (x[:, 1:] > 0)).mean()
        feats.append(float(zc))
        dx = np.diff(x, axis=1)
        mob = float(np.sqrt((dx ** 2).mean() / (np.var(x) + 1e-12)))
        feats.append(mob)
    n = x.shape[1]
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    psd = np.abs(np.fft.rfft(x, axis=1)) ** 2
    cum = np.cumsum(psd.mean(axis=0))
    edge = freqs[int(np.searchsorted(cum, 0.95 * cum[-1]))] if cum[-1] > 0 else 0.0
    feats.append(float(x.shape[0]))
    feats.append(float(edge))
    feats.append(float(np.mean(np.abs(np.diff(x, axis=1)))))
    m4 = float(np.mean((x - x.mean()) ** 4))
    feats.append(float(m4 / ((np.var(x) + 1e-12) ** 2)))
    return np.asarray(feats, dtype=np.float64)


# ------------------------------------------------------------- numpy models

def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def _adam(shape, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(shape) * np.sqrt(2.0 / max(shape[0], 1))
    m = np.zeros(shape)
    v = np.zeros(shape)
    return [p, m, v]


def _adam_step(state, grad, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8,
               l2=0.0):
    p, m, v = state
    grad = grad + l2 * p
    m[:] = b1 * m + (1 - b1) * grad
    v[:] = b2 * v + (1 - b2) * grad * grad
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    p -= lr * mh / (np.sqrt(vh) + eps)
    return p


def fit_softmax(X: np.ndarray, y: np.ndarray, k: int, iters: int = 1500,
                lr: float = 5e-2, l2: float = 1e-4,
                seed: int = 0) -> Dict[str, np.ndarray]:
    """Multinomial logistic regression (full-batch Adam)."""
    n, d = X.shape
    Y = _one_hot(y, k)
    st_w, st_b = _adam((d, k), seed), _adam((k,), seed + 1)
    st_w[0][:] = 0.0
    st_b[0][:] = 0.0
    for t in range(1, iters + 1):
        Z = X @ st_w[0] + st_b[0]
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = X.T @ (P - Y) / n
        Gb = (P - Y).sum(axis=0) / n
        _adam_step(st_w, G, t, lr, l2=l2)
        _adam_step(st_b, Gb, t, lr)
    return {"kind": "softmax", "W": st_w[0], "b": st_b[0]}


def fit_rff_svm(X: np.ndarray, y: np.ndarray, k: int, d_rff: int = 1024,
                iters: int = 1200, lr: float = 1e-2, l2: float = 1e-4,
                seed: int = 0) -> Dict[str, np.ndarray]:
    """RBF-kernel SVM via Random Fourier Features + one-vs-rest hinge."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    dists = np.abs(X[: min(n, 500), None, :] - X[None, : min(n, 500), :])
    med = float(np.median(dists)) + 1e-9
    gamma = 1.0 / (2 * med * med / d)
    W_rff = rng.standard_normal((d, d_rff)) * np.sqrt(2 * gamma)
    b_rff = rng.uniform(0, 2 * np.pi, d_rff)
    Phi = np.sqrt(2.0 / d_rff) * np.cos(X @ W_rff + b_rff)
    Yb = np.where(_one_hot(y, k) > 0.5, 1.0, -1.0)  # one-vs-rest targets
    st_w, st_b = _adam((d_rff, k), seed + 2), _adam((k,), seed + 3)
    st_w[0][:] = 0.0
    st_b[0][:] = 0.0
    for t in range(1, iters + 1):
        M = Phi @ st_w[0] + st_b[0]
        viol = (Yb * M) < 1.0
        G = -(Phi.T @ (Yb * viol)) / n
        Gb = -(Yb * viol).sum(axis=0) / n
        _adam_step(st_w, G, t, lr, l2=l2)
        _adam_step(st_b, Gb, t, lr)
    return {"kind": "rff_svm", "W": st_w[0], "b": st_b[0],
            "rff_w": W_rff, "rff_b": b_rff, "temp": np.array([2.0])}


def fit_mlp(X: np.ndarray, y: np.ndarray, k: int, hidden=(128, 64),
            epochs: int = 60, batch: int = 256, lr: float = 1e-3,
            seed: int = 0) -> Dict[str, list]:
    """2-hidden-layer perceptron (ReLU, minibatch Adam, cross-entropy)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Y = _one_hot(y, k)
    dims = [d, *hidden, k]
    states = [_adam((dims[i], dims[i + 1]), seed + 10 + i)
              for i in range(len(dims) - 1)]
    bs = [_adam((dims[i + 1],), seed + 20 + i)
          for i in range(len(dims) - 1)]
    t = 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            t += 1
            idx = perm[s:s + batch]
            xb, yb = X[idx], Y[idx]
            acts, pres = [xb], []
            for i, (st, bst) in enumerate(zip(states, bs)):
                z = acts[-1] @ st[0] + bst[0]
                pres.append(z)
                acts.append(z if i == len(states) - 1 else np.maximum(z, 0))
            Z = pres[-1] - pres[-1].max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            delta = (P - yb) / len(idx)
            for i in reversed(range(len(states))):
                Gw = acts[i].T @ delta
                Gb = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ states[i][0].T) * (pres[i - 1] > 0)
                _adam_step(states[i], Gw, t, lr, l2=1e-5)
                _adam_step(bs[i], Gb, t, lr)
    return {"kind": "mlp",
            "Ws": [s[0] for s in states], "bs": [b[0] for b in bs]}


def _forward_proba(params: Dict, Xs: np.ndarray) -> np.ndarray:
    kind = params["kind"]
    if kind == "softmax":
        Z = Xs @ params["W"] + params["b"]
    elif kind == "rff_svm":
        Phi = np.sqrt(2.0 / params["rff_w"].shape[1]) * np.cos(
            Xs @ params["rff_w"] + params["rff_b"])
        Z = (Phi @ params["W"] + params["b"]) / float(params["temp"][0])
    elif kind == "mlp":
        A = Xs
        for W, b in zip(params["Ws"][:-1], params["bs"][:-1]):
            A = np.maximum(A @ W + b, 0)
        Z = A @ params["Ws"][-1] + params["bs"][-1]
    else:
        raise ValueError(f"unknown model kind {kind}")
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / (P.sum(axis=1, keepdims=True) + 1e-12)


def standardize_fit(X: np.ndarray) -> Dict[str, np.ndarray]:
    return {"mean": X.mean(axis=0), "scale": X.std(axis=0) + 1e-9}


def standardize_apply(scaler: Dict[str, np.ndarray],
                      X: np.ndarray) -> np.ndarray:
    return (X - scaler["mean"]) / scaler["scale"]


def _window_probas(bundle: Dict, key: str,
                   eeg: np.ndarray) -> Optional[np.ndarray]:
    try:
        feats = np.stack([extract_window_features(w) for w in _windows(eeg)])
        Xs = standardize_apply(bundle["scaler"], feats)
        proba = np.stack([_forward_proba(bundle["models"][key], f[None, :])
                          for f in Xs]).mean(axis=0).ravel()
        proba = proba / (proba.sum() + 1e-12)
        return proba.astype(np.float64)
    except Exception:
        return None


class _NumpyZooClassifier:
    """Shared loader/inference for one NumPy traditional-ML baseline."""

    MODEL_ID: str = ""
    MODEL_KEY: str = ""

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint or os.environ.get(
            f"{self.MODEL_ID.upper()}_CKPT", "")
        bundled = os.path.join(WEIGHTS_DIR, CKPT_NAME)
        if not self.checkpoint and os.path.isfile(bundled):
            self.checkpoint = bundled
        self.bundle: Optional[Dict] = None
        self.checkpoint_loaded = False
        if self.checkpoint and os.path.isfile(self.checkpoint):
            try:
                with open(self.checkpoint, "rb") as fh:
                    bundle = pickle.load(fh)
                if (isinstance(bundle, dict) and "models" in bundle
                        and self.MODEL_KEY in bundle["models"]):
                    self.bundle = bundle
                    self.checkpoint_loaded = True
            except Exception:
                self.bundle = None
                self.checkpoint_loaded = False

    def predict(self, eeg: np.ndarray, **_) -> Dict:
        if self.bundle is not None:
            proba = _window_probas(self.bundle, self.MODEL_KEY, eeg)
            if proba is not None and proba.shape == (len(TARGET_CLASSES),):
                logits = np.log(proba + 1e-12)
                return _format_prediction(
                    logits, self.MODEL_ID,
                    f"numpy-{self.MODEL_KEY} (checkpoint)",
                    used_fallback=False)
        return _format_prediction(
            _numpy_fallback_logits(eeg), self.MODEL_ID,
            f"numpy-{self.MODEL_KEY} (random init, no checkpoint)",
            used_fallback=True)


class LogisticRegressionClassifier(_NumpyZooClassifier):
    MODEL_ID = "logistic_regression"
    MODEL_KEY = "logreg"


class SVMClassifier(_NumpyZooClassifier):
    MODEL_ID = "svm"
    MODEL_KEY = "svm"


class MLPClassifierZoo(_NumpyZooClassifier):
    MODEL_ID = "mlp"
    MODEL_KEY = "mlp"

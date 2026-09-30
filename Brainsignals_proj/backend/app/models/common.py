"""Shared model-zoo constants, geometry helpers, and inference mixin.

Fixed input geometry for all deep models: max native channel count in the
dataset (31; the 25-ch Air Conditioner layout is zero-padded, never
interpolated) and a 12 s window @120 Hz. Full files are covered by
multi-window logit averaging at inference.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

TARGET_CLASSES: List[str] = [
    "Air Conditioner",
    "Bluetooth Speaker",
    "Door Lock",
    "Electric Light",
    "TV",
]

N_CHANNELS = 31
WIN_SAMPLES = 1440

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

MODEL_CATALOG: List[Dict[str, str]] = [
    {"id": "eeg_tcfnet", "name": "EEG-TCFNet (Proposed Base Model)"},
    {"id": "eeg_tcnet_lstm", "name": "EEG-TCNet + LSTM"},
    {"id": "eeg_tcnet", "name": "EEG-TCNet (EEGNet + TCN)"},
    {"id": "lenet", "name": "LeNet (2D CNN Baseline)"},
    {"id": "logistic_regression", "name": "Logistic Regression"},
    {"id": "svm", "name": "SVM (RBF Baseline)"},
    {"id": "mlp", "name": "MLP Classifier"},
]
DEFAULT_MODEL = "eeg_tcfnet"
VALID_MODEL_IDS = [m["id"] for m in MODEL_CATALOG]

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    HAS_TORCH = False


def _fit_channels(seg: np.ndarray, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Zero-pad (or truncate) the channel axis to the fixed model layout."""
    ch, t = seg.shape
    if ch == n_channels:
        return np.asarray(seg, dtype=np.float64)
    if ch > n_channels:
        return np.asarray(seg[:n_channels, :], dtype=np.float64)
    out = np.zeros((n_channels, t), dtype=np.float64)
    out[:ch, :] = seg
    return out


def _windows(eeg: np.ndarray, win: int = WIN_SAMPLES,
             max_windows: int = 8) -> List[np.ndarray]:
    """Evenly-spaced windows covering the whole file (time-pad if short)."""
    _, t = eeg.shape
    if t <= win:
        return [np.pad(eeg, ((0, 0), (0, win - t)))]
    if t <= win + max_windows - 1:
        starts = list(range(t - win + 1))[:max_windows]
    else:
        starts = np.linspace(0, t - win, max_windows).round().astype(int).tolist()
    return [eeg[:, s:s + win] for s in starts]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def _numpy_fallback_logits(eeg: np.ndarray) -> np.ndarray:
    """Deterministic fallback: ERP-band energy fingerprint -> logits."""
    ch, t = eeg.shape
    x = eeg - eeg.mean(axis=1, keepdims=True)
    seg = x[:, : min(t, 2400)]
    energy = np.array([float(np.mean(p ** 2) + 1e-9)
                       for p in np.array_split(seg, 5, axis=1)])
    idx = np.array_split(np.arange(ch), 5)
    spat = np.array([float(np.mean(seg[g, :] ** 2) + 1e-9) if len(g) else 1e-9
                     for g in idx])
    feat = np.log(energy + 1e-9) + 0.5 * np.log(spat + 1e-9)
    feat = (feat - feat.mean()) / (feat.std() + 1e-9)
    return (feat * 2.0).astype(np.float64)


def _format_prediction(logits: np.ndarray, model_id: str, source: str,
                       used_fallback: bool, has_torch: bool = False) -> Dict:
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    cid0 = int(np.argmax(probs))
    return {
        "predicted_appliance": TARGET_CLASSES[cid0],
        "class_id": cid0 + 1,  # 1-indexed, dataset convention
        "confidence_scores": {c: round(float(p), 4)
                              for c, p in zip(TARGET_CLASSES, probs)},
        "logits": [round(float(v), 4) for v in np.asarray(logits).ravel()],
        "used_fallback": used_fallback,
        "has_torch": has_torch,
        "model_used": model_id,
        "model_source": source,
    }


class TorchZooClassifier:
    """Mixin: checkpoint auto-load + multi-window mean-logit inference.

    Subclasses set ``MODEL_ID`` and ``_build_model()``. Checkpoint resolution:
    explicit arg -> ``<MODEL_ID upper>_CKPT`` env -> bundled weights file.
    """

    MODEL_ID: str = ""
    CKPT_NAME: str = ""

    def _build_model(self):  # -> torch.nn.Module
        raise NotImplementedError

    def _ckpt_candidates(self) -> List[str]:
        cands = []
        env = os.environ.get(f"{self.MODEL_ID.upper()}_CKPT", "")
        if env:
            cands.append(env)
        if self.CKPT_NAME:
            cands.append(os.path.join(WEIGHTS_DIR, self.CKPT_NAME))
        return cands

    def __init__(self, checkpoint: Optional[str] = None):
        from torch import nn as _nn  # noqa: F401  (guarded by HAS_TORCH below)
        self.checkpoint = checkpoint or ""
        if not self.checkpoint:
            for c in self._ckpt_candidates():
                if c and os.path.isfile(c):
                    self.checkpoint = c
                    break
        self.model = None
        self.checkpoint_loaded = False
        if HAS_TORCH:
            try:
                self.model = self._build_model()
                self.model.eval()
                if self.checkpoint and os.path.isfile(self.checkpoint):
                    import torch as _torch

                    state = _torch.load(self.checkpoint, map_location="cpu")
                    self.model.load_state_dict(state.get("model", state))
                    self.model.eval()
                    self.checkpoint_loaded = True
            except Exception:
                self.model = None
                self.checkpoint_loaded = False

    def _torch_logits(self, eeg: np.ndarray) -> Optional[np.ndarray]:
        if not HAS_TORCH or self.model is None:
            return None
        try:
            import torch as _torch

            seg = _fit_channels(eeg, N_CHANNELS)
            wins = _windows(seg)
            with _torch.no_grad():
                batch = _torch.tensor(
                    np.stack([w[np.newaxis, :, :] for w in wins]),
                    dtype=_torch.float32)  # (W, 1, C, T)
                logits = self.model(batch).cpu().numpy()
            mean_logits = logits.mean(axis=0).ravel()
            if mean_logits.shape != (len(TARGET_CLASSES),):
                return None
            return mean_logits.astype(np.float64)
        except Exception:
            return None

    def predict(self, eeg: np.ndarray, **_) -> Dict:
        logits = self._torch_logits(eeg)
        if logits is not None:
            source = (f"torch-{self.MODEL_ID}"
                      + (" (checkpoint)" if self.checkpoint_loaded
                         else " (random init, no checkpoint)"))
            return _format_prediction(logits, self.MODEL_ID, source,
                                      used_fallback=not self.checkpoint_loaded,
                                      has_torch=True)
        return _format_prediction(
            _numpy_fallback_logits(eeg), self.MODEL_ID,
            (f"torch-{self.MODEL_ID} (random init, no checkpoint)" if HAS_TORCH
             else "numpy-fallback (torch not installed)"),
            used_fallback=True, has_torch=HAS_TORCH)

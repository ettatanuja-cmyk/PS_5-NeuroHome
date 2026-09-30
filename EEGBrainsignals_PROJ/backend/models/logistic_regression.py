"""Multinomial Logistic Regression baseline (16-feature, fast CPU).

Refactored wrapper around the existing ``appliance_clf.npz`` checkpoint
(trained by ``backend/train_appliance_clf.py``): multinomial logistic
regression on 16 per-file dataset features (geometry + trigger/paradigm +
amplitude stats + spectral shape). Pure NumPy inference, no torch needed.

Exposes the same ``predict(eeg, eeg_raw, trigger, orig_fs)`` interface as
the deep models so ``main.py`` can route to it via ``model_name``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

MODEL_ID = "logistic_regression"
MODEL_NAME = "Multinomial Logistic Regression (16-Feature Baseline)"
TARGET_CLASSES: List[str] = [
    "Air Conditioner",
    "Bluetooth Speaker",
    "Door Lock",
    "Electric Light",
    "TV",
]

HAS_TORCH = False  # CPU-only baseline by design


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def _numpy_fallback_logits(eeg: np.ndarray) -> np.ndarray:
    """Deterministic fallback when the trained checkpoint is missing."""
    ch, t = eeg.shape
    x = eeg - eeg.mean(axis=1, keepdims=True)
    win = min(t, 2400)
    seg = x[:, :win]
    parts = np.array_split(seg, 5, axis=1)
    energy = np.array([float(np.mean(p ** 2) + 1e-9) for p in parts])
    idx = np.array_split(np.arange(ch), 5)
    spat = np.array(
        [float(np.mean(seg[g, :] ** 2) + 1e-9) if len(g) else 1e-9 for g in idx]
    )
    feat = np.log(energy + 1e-9) + 0.5 * np.log(spat + 1e-9)
    feat = (feat - feat.mean()) / (feat.std() + 1e-9)
    return (feat * 2.0).astype(np.float64)


class LogisticRegressionClassifier:
    """Fast CPU baseline using the trained 16-feature logreg checkpoint."""

    def __init__(self, checkpoint: str | None = None):
        self._clf = None
        self.checkpoint_loaded = False
        try:
            from services.appliance_clf import TrainedApplianceClf

            self._clf = (
                TrainedApplianceClf(path=checkpoint)
                if checkpoint
                else TrainedApplianceClf()
            )
            self.checkpoint_loaded = bool(
                getattr(self._clf, "available", False)
            )
        except Exception:
            self._clf = None
            self.checkpoint_loaded = False

    def predict(
        self,
        eeg: np.ndarray,
        eeg_raw=None,
        trigger=None,
        orig_fs: float = 500.0,
    ) -> Dict:
        # 1) trained ML checkpoint on dataset features (preferred)
        if eeg_raw is not None and self._clf is not None:
            try:
                r = self._clf.predict_proba(eeg_raw, trigger, orig_fs)
                if r is not None:
                    return {
                        "predicted_appliance": r["predicted"],
                        "class_id": r["class_id"],
                        "confidence_scores": {
                            c: round(float(r["probs"].get(c, 0.0)), 4)
                            for c in TARGET_CLASSES
                        },
                        "logits": [],
                        "used_fallback": False,
                        "has_torch": False,
                        "model_source": (
                            "trained-ml (logreg on 16 dataset features)"
                        ),
                    }
            except Exception:
                pass
        # 2) deterministic fallback (checkpoint missing/unusable)
        eeg = np.asarray(eeg, dtype=np.float64)
        logits = _numpy_fallback_logits(eeg)
        probs = _softmax(logits)
        cid0 = int(np.argmax(probs))
        return {
            "predicted_appliance": TARGET_CLASSES[cid0],
            "class_id": cid0 + 1,
            "confidence_scores": {
                c: round(float(p), 4) for c, p in zip(TARGET_CLASSES, probs)
            },
            "logits": [round(float(v), 4) for v in logits],
            "used_fallback": True,
            "has_torch": False,
            "model_source": "numpy-fallback (no trained weights)",
        }

"""Trained appliance classifier: loads models/appliance_clf.npz.

Multinomial logistic regression on the 16 shared features
(services/appliance_features.py). Pure NumPy inference.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from services.appliance_features import extract_features

_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "models", "appliance_clf.npz")


class TrainedApplianceClf:
    def __init__(self, path: str = _CKPT):
        self.path = path
        self.classes: List[str] = []
        self.coef: Optional[np.ndarray] = None      # (K, F)
        self.intercept: Optional[np.ndarray] = None  # (K,)
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.available = False
        try:
            if os.path.isfile(path):
                z = np.load(path, allow_pickle=False)
                self.classes = [str(c) for c in z["classes"]]
                self.coef = np.asarray(z["coef_"], dtype=np.float64)
                self.intercept = np.asarray(z["intercept_"], dtype=np.float64)
                self.mean = np.asarray(z["mean_"], dtype=np.float64)
                self.scale = np.asarray(z["scale_"], dtype=np.float64)
                self.available = True
        except Exception:
            self.available = False

    def predict_proba(self, eeg_raw: np.ndarray, trigger=None,
                      orig_fs: float = 500.0) -> Optional[Dict]:
        if not self.available:
            return None
        try:
            x = extract_features(eeg_raw, trigger, orig_fs)
            xs = (x - self.mean) / self.scale
            z = self.coef @ xs + self.intercept
            z -= z.max()
            e = np.exp(z)
            p = e / (e.sum() + 1e-12)
            cid = int(np.argmax(p))
            return {
                "class_id": cid + 1,
                "predicted": self.classes[cid],
                "probs": {c: float(v) for c, v in zip(self.classes, p)},
            }
        except Exception:
            return None

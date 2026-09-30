"""EEGNet / Shallow ConvNet for ERP appliance classification.

Standard EEGNet architecture (Lawhern et al. 2018): temporal convolution
followed by depthwise spatial convolution and a separable convolution block
(ELU + average pooling), then a dense softmax head.

Input tensor: (batch_size, 1, n_channels, n_samples).

Torch is optional: without torch (or without a checkpoint) inference falls
back to a deterministic NumPy ERP-energy fingerprint so the API always
returns a valid 5-way prediction.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

MODEL_ID = "eegnet"
MODEL_NAME = "EEGNet (Spatial-Temporal Conv)"
TARGET_CLASSES: List[str] = [
    "Air Conditioner",
    "Bluetooth Speaker",
    "Door Lock",
    "Electric Light",
    "TV",
]

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # torch optional -> numpy fallback
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False


# ------------------------------------------------------------ torch model ---

if HAS_TORCH:

    class EEGNet(nn.Module):
        """Standard EEGNet: temporal -> depthwise spatial -> separable convs."""

        def __init__(
            self,
            n_channels: int = 25,
            n_classes: int = 5,
            f1: int = 8,
            d: int = 2,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.n_channels = n_channels
            f2 = f1 * d
            # Block 1: temporal conv + depthwise spatial conv
            self.temporal = nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
            self.bn1 = nn.BatchNorm2d(f1)
            self.depthwise = nn.Conv2d(
                f1, f2, (n_channels, 1), groups=f1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(f2)
            self.elu = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)
            # Block 2: separable convolution (depthwise + pointwise)
            self.separable_dw = nn.Conv2d(
                f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False
            )
            self.separable_pw = nn.Conv2d(f2, f2, (1, 1), bias=False)
            self.bn3 = nn.BatchNorm2d(f2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)
            # Fixed-size pooling -> dense head (time/length agnostic)
            self.adapt_pool = nn.AdaptiveAvgPool2d((1, 8))
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(),
                nn.Linear(f2 * 8, n_classes),
            )

        def forward(self, x):  # x: (B, 1, C, T)
            x = self.elu(self.bn1(self.temporal(x)))
            x = self.elu(self.bn2(self.depthwise(x)))
            x = self.drop1(self.pool1(x))
            x = self.elu(
                self.bn3(self.separable_pw(self.separable_dw(x)))
            )
            x = self.drop2(self.pool2(x))
            x = self.adapt_pool(x)
            return self.head(x)  # logits (B, n_classes)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def _numpy_fallback_logits(eeg: np.ndarray) -> np.ndarray:
    """Deterministic fallback: per-channel ERP-band energy distribution."""
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


class EEGNetClassifier:
    """Inference wrapper with optional torch checkpoint.

    Env checkpoint override: ``EEGNET_CKPT=/path/to/model.pth``.
    """

    def __init__(self, checkpoint: Optional[str] = None, n_channels: int = 25):
        self.checkpoint = checkpoint or os.environ.get("EEGNET_CKPT", "")
        self.model = None
        self.checkpoint_loaded = False
        if HAS_TORCH:
            try:
                self.model = EEGNet(n_channels=n_channels)
                if self.checkpoint and os.path.isfile(self.checkpoint):
                    state = torch.load(self.checkpoint, map_location="cpu")
                    self.model.load_state_dict(state.get("model", state))
                    self.checkpoint_loaded = True
                self.model.eval()
            except Exception:
                self.model = None
                self.checkpoint_loaded = False

    def _torch_logits(self, eeg: np.ndarray) -> Optional[np.ndarray]:
        if not HAS_TORCH or self.model is None:
            return None
        try:
            ch, t = eeg.shape
            t_in = min(max(t, 128), 2048)
            if t > t_in:  # center crop
                s = (t - t_in) // 2
                seg = eeg[:, s : s + t_in]
            else:
                seg = np.pad(eeg, ((0, 0), (0, t_in - t)))
            n_ch = self.model.n_channels
            if ch != n_ch:  # interpolate channel axis to trained layout
                src = np.linspace(0, 1, ch)
                dst = np.linspace(0, 1, n_ch)
                seg = np.stack(
                    [np.interp(dst, src, seg[:, j]) for j in range(seg.shape[1])],
                    axis=1,
                )
            with torch.no_grad():
                xt = torch.tensor(
                    seg[np.newaxis, np.newaxis, :, :], dtype=torch.float32
                )
                logits = self.model(xt).cpu().numpy().ravel()
            return logits.astype(np.float64)
        except Exception:
            return None

    def predict(
        self,
        eeg: np.ndarray,
        eeg_raw=None,
        trigger=None,
        orig_fs: float = 500.0,
    ) -> Dict:
        # 1) Prefer the trained 16-feature logreg checkpoint whenever the raw
        # file is available: without a torch checkpoint the deep net is
        # untrained (random weights / numpy fingerprint) and misclassifies.
        if eeg_raw is not None:
            try:
                from services.appliance_clf import TrainedApplianceClf

                _clf = TrainedApplianceClf()
                if _clf.available:
                    r = _clf.predict_proba(eeg_raw, trigger, orig_fs)
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
                            "has_torch": HAS_TORCH,
                            "model_source": (
                                "trained-ml (logreg on 16 dataset features) via eegnet"
                                + (" + torch-eegnet (no checkpoint)" if HAS_TORCH else "")
                            ),
                        }
            except Exception:
                pass
        logits = self._torch_logits(np.asarray(eeg, dtype=np.float64))
        used_fallback = False
        if logits is None or logits.shape != (5,):
            logits = _numpy_fallback_logits(np.asarray(eeg, dtype=np.float64))
            used_fallback = True
        if self.checkpoint_loaded and not used_fallback:
            source = "torch-eegnet (trained checkpoint)"
        elif not used_fallback:
            source = "torch-eegnet (random init, no checkpoint)"
        else:
            source = "numpy-fallback (no torch/checkpoint)"
        probs = _softmax(logits)
        cid0 = int(np.argmax(probs))
        return {
            "predicted_appliance": TARGET_CLASSES[cid0],
            "class_id": cid0 + 1,
            "confidence_scores": {
                c: round(float(p), 4) for c, p in zip(TARGET_CLASSES, probs)
            },
            "logits": [round(float(v), 4) for v in logits],
            "used_fallback": used_fallback,
            "has_torch": HAS_TORCH,
            "model_source": source,
        }

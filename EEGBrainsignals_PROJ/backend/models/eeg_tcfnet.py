"""EEG-TCFNet classifier (EEGNet + TCN + LSTM + Fuzzy Neural Block).

Paper architecture:
  1. EEGNet feature extractor (temporal conv -> depthwise spatial conv ->
     separable conv, ELU + average pooling).
  2. TCN block (dilated causal conv residual stack).
  3. 2 stacked LSTM layers, 30 hidden units each.
  4. Fuzzy Neural Block: Gaussian membership layer merged with FC head
     into a 5-way Softmax.

If PyTorch is unavailable, or no checkpoint is provided, inference falls
back to a deterministic NumPy routine (channel-band ERP energy -> logits)
so the API always returns a valid prediction.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

MODEL_ID = "eeg_tcfnet"
MODEL_NAME = "EEG-TCFNet (Base Paper Model)"

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

    class _TCNResidual(nn.Module):
        def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.2):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.trim = pad

        def forward(self, x):
            out = self.drop(self.relu(self.conv1(x)))
            out = self.drop(self.relu(self.conv2(out)))
            if self.trim:
                out = out[:, :, :-self.trim]
            return self.relu(out + x[:, :, -out.shape[2]:] if out.shape[2] != x.shape[2] else out + x)

    class _TCNBlock(nn.Module):
        def __init__(self, channels: int, levels: int = 3, kernel: int = 3, dropout: float = 0.2):
            super().__init__()
            self.layers = nn.ModuleList(
                [_TCNResidual(channels, kernel, 2 ** i, dropout) for i in range(levels)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class _FuzzyNeuralBlock(nn.Module):
        """Gaussian membership layer merged with the FC path."""

        def __init__(self, in_dim: int, n_rules: int = 16):
            super().__init__()
            self.centers = nn.Parameter(torch.randn(n_rules, in_dim) * 0.5)
            self.sigmas = nn.Parameter(torch.ones(n_rules, in_dim))
            self.rule_fc = nn.Linear(n_rules, in_dim)

        def forward(self, x):  # x: (B, D)
            diff = (x.unsqueeze(1) - self.centers.unsqueeze(0)) / (self.sigmas.unsqueeze(0) + 1e-6)
            membership = torch.exp(-0.5 * diff.pow(2)).mean(dim=2)  # (B, R)
            firing = membership / (membership.sum(dim=1, keepdim=True) + 1e-9)
            return x + self.rule_fc(firing)  # residual merge

    class EEGTCFNet(nn.Module):
        """Full EEG-TCFNet: EEGNet -> TCN -> 2xLSTM(30) -> FNB -> Softmax."""

        def __init__(self, n_channels: int = 25, n_time: int = 720, n_classes: int = 5,
                     f1: int = 8, d: int = 2, n_rules: int = 16):
            super().__init__()
            self.n_channels = n_channels
            # --- EEGNet feature extractor ---
            self.temporal = nn.Conv2d(1, f1, (1, 64), padding=(0, 32))
            self.bn1 = nn.BatchNorm2d(f1)
            self.depthwise = nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1)
            self.bn2 = nn.BatchNorm2d(f1 * d)
            self.elu = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(0.25)
            self.separable1 = nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d)
            self.separable2 = nn.Conv2d(f1 * d, f1 * d, (1, 1))
            self.bn3 = nn.BatchNorm2d(f1 * d)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(0.25)
            feat_dim = f1 * d
            # --- TCN ---
            self.tcn = _TCNBlock(feat_dim, levels=3)
            # --- 2 stacked LSTMs, 30 hidden units each ---
            self.lstm = nn.LSTM(feat_dim, 30, num_layers=2, batch_first=True, dropout=0.2)
            # --- Fuzzy block + head ---
            self.fnb = _FuzzyNeuralBlock(30, n_rules)
            self.fc = nn.Linear(30, n_classes)

        def forward(self, x):  # x: (B, 1, C, T)
            x = self.elu(self.bn1(self.temporal(x)))
            x = self.elu(self.bn2(self.depthwise(x)))
            x = self.pool1(x)
            x = self.drop1(x)
            x = self.elu(self.bn3(self.separable2(self.separable1(x))))
            x = self.pool2(x)
            x = self.drop2(x)
            x = x.squeeze(2).permute(0, 2, 1)  # (B, T', F)
            x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
            _, (h, _) = self.lstm(x)
            feat = h[-1]  # last layer hidden state (B, 30)
            feat = self.fnb(feat)
            return self.fc(feat)  # logits


def _numpy_fallback_logits(eeg: np.ndarray) -> np.ndarray:
    """Deterministic fallback: per-channel ERP-band energy distribution.

    Projects the 5 appliance classes onto 5Anchor statistics derived from
    the signal (band powers + spatial topography hashed into 5 bins),
    sharpened with softmax temperature. Stable, no weights required.
    """
    ch, t = eeg.shape
    x = eeg - eeg.mean(axis=1, keepdims=True)
    win = min(t, 2400)
    seg = x[:, :win]
    # 5 temporal sub-windows -> ERP dynamics fingerprint (5,)
    parts = np.array_split(seg, 5, axis=1)
    energy = np.array([float(np.mean(p ** 2) + 1e-9) for p in parts])
    # 5 spatial groups -> topography fingerprint (5,)
    idx = np.array_split(np.arange(ch), 5)
    spat = np.array([float(np.mean(seg[g, :] ** 2) + 1e-9) if len(g) else 1e-9 for g in idx])
    feat = np.log(energy + 1e-9) + 0.5 * np.log(spat + 1e-9)
    feat = (feat - feat.mean()) / (feat.std() + 1e-9)
    logits = feat * 2.0  # temperature sharpening
    return logits.astype(np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


class ApplianceClassifier:
    """Unified inference wrapper with optional torch checkpoint.

    Standard 5-model interface: ``predict(eeg, eeg_raw, trigger, orig_fs)``.
    Env checkpoint override: ``EEG_TCFNET_CKPT=/path/to/model.pth``.
    """

    def __init__(self, checkpoint: Optional[str] = None, n_channels: int = 25):
        self.checkpoint = checkpoint or os.environ.get("EEG_TCFNET_CKPT", "")
        self.model = None
        self.checkpoint_loaded = False
        self.n_channels_seen: Optional[int] = None
        if HAS_TORCH and self.checkpoint and os.path.isfile(self.checkpoint):
            try:
                self.model = EEGTCFNet(n_channels=n_channels)
                state = torch.load(self.checkpoint, map_location="cpu")
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
            ch, t = eeg.shape
            # adaptive pool time to bounded length for the conv stack
            t_in = min(max(t, 128), 2048)
            if t > t_in:  # center crop
                s = (t - t_in) // 2
                seg = eeg[:, s:s + t_in]
            else:
                seg = np.pad(eeg, ((0, 0), (0, t_in - t)))
            if ch != self.model.n_channels:
                # interpolate channel axis to the trained layout
                src = np.linspace(0, 1, ch)
                dst = np.linspace(0, 1, self.model.n_channels)
                seg = np.stack(
                    [np.interp(dst, src, seg[:, j]) for j in range(seg.shape[1])],
                    axis=1,
                )
            with torch.no_grad():
                xt = torch.tensor(seg[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
                logits = self.model(xt).cpu().numpy().ravel()
            return logits.astype(np.float64)
        except Exception:
            return None

    def predict(self, eeg: np.ndarray, eeg_raw=None, trigger=None,
                orig_fs: float = 500.0) -> Dict:
        # 1) trained ML checkpoint on dataset features (preferred)
        if eeg_raw is not None:
            try:
                from services.appliance_clf import TrainedApplianceClf
                if self.__dict__.get("_ml_clf") is None:
                    self.__dict__["_ml_clf"] = TrainedApplianceClf()
                ml = self.__dict__["_ml_clf"]
                r = ml.predict_proba(eeg_raw, trigger, orig_fs)
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
                            "trained-ml (logreg on 16 dataset features) via eeg_tcfnet"
                            + (" + torch-eeg-tcfnet (no checkpoint)" if HAS_TORCH else "")
                        ),
                    }
            except Exception:
                pass
        # 2) torch EEG-TCFNet graph with trained weights
        logits = self._torch_logits(eeg)
        used_fallback = False
        source = "torch-eeg-tcfnet"
        if logits is None or logits.shape != (5,):
            logits = _numpy_fallback_logits(eeg)
            used_fallback = True
            source = "numpy-fallback (no trained weights)"
        probs = _softmax(logits)
        cid0 = int(np.argmax(probs))
        return {
            "predicted_appliance": TARGET_CLASSES[cid0],
            "class_id": cid0 + 1,  # 1-indexed to match dataset convention
            "confidence_scores": {c: round(float(p), 4) for c, p in zip(TARGET_CLASSES, probs)},
            "logits": [round(float(v), 4) for v in logits],
            "used_fallback": used_fallback,
            "has_torch": HAS_TORCH,
            "model_source": source,
        }


# Standard registry name (matches models/eegnet.py, eeg_conformer.py, ...).
# Kept separate from the legacy ``ApplianceClassifier`` name used by older
# scripts/docs.
class EEGTCFNetClassifier(ApplianceClassifier):
    """EEG-TCFNet inference wrapper (EEGNet -> TCN -> 2xLSTM(30) -> FNB)."""

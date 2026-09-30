"""EEG-TCFNet (proposed base model): EEG-TCNet + LSTM + Fuzzy Neural Block.

  * EEG-Net core: Conv2D (1x20) -> Depthwise Conv2D (16x1) -> Separable
    Conv2D, ELU activation, AveragePooling (1x4 and 1x5), Dropout (0.5).
  * TCN: temporal convolutions (kernel 2x2, 6 filters).
  * LSTM block: two stacked LSTM layers, 30 hidden units each.
  * Fuzzy Neural Block (FNB): Gaussian membership function
      mu_i^k(v_i, c^k, a) = exp(-1/4 * (v_i - c^k)^2 / a^2)
    with trainable centroids c^k (FCM-seeded, updated per epoch) and a
    learnable scale parameter a.
  * Dual-path fusion: Dense (fully connected) path + FNB path merged before
    the final Softmax activation layer.

Checkpoint: ``weights/eeg_tcfnet.pth``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .common import HAS_TORCH, N_CHANNELS, TARGET_CLASSES, TorchZooClassifier

if HAS_TORCH:
    import torch
    import torch.nn as nn

    class FuzzyNeuralBlock(nn.Module):
        """Fuzzy Neural Block with the paper membership function.

        mu_i^k(v_i, c^k, a) = exp(-1/4 * (v_i - c^k)^2 / a^2)
        """

        def __init__(self, in_dim: int, n_rules: int = 16, a_init: float = 1.0):
            super().__init__()
            self.centers = nn.Parameter(torch.randn(n_rules, in_dim) * 0.5)
            self.log_a = nn.Parameter(
                torch.full((n_rules, in_dim), float(a_init)).log().clamp(-4, 4))
            self.rule_fc = nn.Linear(n_rules, in_dim)

        @torch.no_grad()
        def init_centers_fcm(self, data: "torch.Tensor", m: float = 2.0,
                             iters: int = 20) -> None:
            """Seed centroids c^k with fuzzy c-means clustering of a batch."""
            x = data.detach().reshape(-1, data.shape[-1]).float()
            n = x.shape[0]
            r = self.centers.shape[0]
            u = torch.rand(n, r)
            u = u / u.sum(1, keepdim=True)
            for _ in range(iters):
                um = u.pow(m)
                centers = (um.T @ x) / (um.sum(0, keepdim=True).T + 1e-9)
                dist = ((x[:, None, :] - centers[None]) ** 2).sum(-1) + 1e-9
                inv = (1.0 / dist).pow(2.0 / (m - 1.0))
                u = inv / inv.sum(1, keepdim=True)
            self.centers.copy_(centers[:r].to(self.centers.dtype))

        def membership(self, v: "torch.Tensor") -> "torch.Tensor":
            a = nn.functional.softplus(self.log_a) + 1e-6
            diff = (v.unsqueeze(1) - self.centers.unsqueeze(0)) / a.unsqueeze(0)
            return torch.exp(-0.25 * diff.pow(2))  # exact paper formula

        def forward(self, v: "torch.Tensor") -> "torch.Tensor":
            mu = self.membership(v)                    # (B, R, D)
            firing = mu.mean(dim=2)                    # (B, R)
            firing = firing / (firing.sum(1, keepdim=True) + 1e-9)
            return self.rule_fc(firing)                # (B, D)

    class EEGTCFNet(nn.Module):
        """Full EEG-TCFNet: EEGNet -> TCN -> 2xLSTM(30) -> FNB -> Softmax."""

        def __init__(self, n_channels: int = N_CHANNELS,
                     n_classes: int = len(TARGET_CLASSES), n_rules: int = 16):
            super().__init__()
            self.n_channels = n_channels
            # --- EEG-Net feature extractor ---
            self.conv_temporal = nn.Conv2d(1, 8, (1, 20), padding=(0, 10))
            self.bn1 = nn.BatchNorm2d(8)
            self.conv_depthwise = nn.Conv2d(8, 16, (n_channels, 1), groups=8)
            self.bn2 = nn.BatchNorm2d(16)
            self.elu = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(0.5)
            self.conv_separable_dw = nn.Conv2d(16, 16, (1, 16),
                                               padding=(0, 8), groups=16)
            self.conv_separable_pw = nn.Conv2d(16, 16, (1, 1))
            self.bn3 = nn.BatchNorm2d(16)
            self.pool2 = nn.AvgPool2d((1, 5))
            self.drop2 = nn.Dropout(0.5)
            # --- TCN: temporal conv, kernel 2x2, 6 filters ---
            self.tcn = nn.Conv2d(16, 6, (1, 2), padding=(0, 1))
            self.tcn_bn = nn.BatchNorm2d(6)
            # --- two stacked LSTMs, 30 hidden units each ---
            self.lstm = nn.LSTM(6, 30, num_layers=2, batch_first=True)
            # --- dual-path fusion: dense path + fuzzy path ---
            self.dense_path = nn.Linear(30, 30)
            self.fnb = FuzzyNeuralBlock(30, n_rules)
            self.classifier = nn.Linear(60, n_classes)

        def forward(self, x):  # x: (B, 1, C, T)
            x = self.elu(self.bn1(self.conv_temporal(x)))
            x = self.elu(self.bn2(self.conv_depthwise(x)))
            x = self.drop1(self.pool1(x))
            x = self.elu(self.bn3(self.conv_separable_pw(
                self.conv_separable_dw(x))))
            x = self.drop2(self.pool2(x))
            x = self.elu(self.tcn_bn(self.tcn(x)))
            x = x.squeeze(2).permute(0, 2, 1)      # (B, T', 6)
            _, (h, _) = self.lstm(x)
            feat = h[-1]                            # (B, 30)
            fused = torch.cat([self.dense_path(feat), self.fnb(feat)], dim=1)
            return self.classifier(fused)           # logits -> Softmax outside


class EEGTCFNetClassifier(TorchZooClassifier):
    MODEL_ID = "eeg_tcfnet"
    CKPT_NAME = "eeg_tcfnet.pth"

    def _build_model(self):
        return EEGTCFNet(n_channels=N_CHANNELS,
                         n_classes=len(TARGET_CLASSES))

    def predict(self, eeg: np.ndarray, **_) -> Dict:
        return super().predict(np.asarray(eeg, dtype=np.float64))

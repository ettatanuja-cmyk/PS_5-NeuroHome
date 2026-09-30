"""LeNet baseline (base paper): simple 2D convolutional network.

Conv2D -> AveragePooling -> ReLU -> Conv2D -> AveragePooling -> ReLU ->
FC(120) -> FC(84) -> Softmax(5). An adaptive average-pool bounds the
flattened size for the 31x1440 EEG window input. Checkpoint:
``weights/lenet.pth``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .common import HAS_TORCH, N_CHANNELS, TorchZooClassifier

MODEL_ID = "lenet"
CKPT_NAME = "lenet.pth"

if HAS_TORCH:
    import torch
    import torch.nn as nn

    class LeNet(nn.Module):
        """LeNet-5 style network adapted to (1, 31, 1440) EEG windows."""

        def __init__(self, n_classes: int = 5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, (5, 5)),
                nn.AvgPool2d((2, 2)),
                nn.ReLU(),
                nn.Conv2d(6, 16, (5, 5)),
                nn.AvgPool2d((2, 2)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 30)),  # (16, 4, 30) = 1920
            )
            self.classifier = nn.Sequential(
                nn.Linear(16 * 4 * 30, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, n_classes),
            )

        def forward(self, x):  # x: (B, 1, C, T)
            if x.shape[2] != N_CHANNELS:  # defensive channel fit
                if x.shape[2] < N_CHANNELS:
                    pad = x.new_zeros(x.shape[0], 1,
                                      N_CHANNELS - x.shape[2], x.shape[3])
                    x = torch.cat([x, pad], dim=2)
                else:
                    x = x[:, :, :N_CHANNELS, :]
            return self.classifier(self.features(x).flatten(1))


class LeNetClassifier(TorchZooClassifier):
    MODEL_ID = MODEL_ID
    CKPT_NAME = CKPT_NAME

    def _build_model(self):
        return LeNet(n_classes=5)

    def predict(self, eeg: np.ndarray, **_) -> Dict:
        return super().predict(np.asarray(eeg, dtype=np.float64))

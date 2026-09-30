"""EEG-TCNet architectures (base paper).

  * EEG-TCNet: spatial filtering (EEGNet core) + 1D/2D Temporal Convolutional
    Network (TCN) -> Softmax.
  * EEG-TCNet + LSTM: EEG-TCNet augmented with two sequential LSTM layers
    (30 hidden units each) -> Softmax.

Checkpoints: ``weights/eeg_tcnet.pth``, ``weights/eeg_tcnet_lstm.pth``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .common import HAS_TORCH, N_CHANNELS, TARGET_CLASSES, TorchZooClassifier

if HAS_TORCH:
    import torch.nn as nn

    class EEGNetCore(nn.Module):
        """Spatial filtering front-end: temporal conv -> depthwise spatial
        conv -> separable conv, ELU + average pooling."""

        def __init__(self, n_channels: int = N_CHANNELS, f1: int = 8, d: int = 2):
            super().__init__()
            self.t_conv = nn.Conv2d(1, f1, (1, 20), padding=(0, 10))
            self.t_bn = nn.BatchNorm2d(f1)
            self.spatial_dw = nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1)
            self.s_bn = nn.BatchNorm2d(f1 * d)
            self.elu = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(0.5)
            self.sep_dw = nn.Conv2d(f1 * d, f1 * d, (1, 16),
                                    padding=(0, 8), groups=f1 * d)
            self.sep_pw = nn.Conv2d(f1 * d, f1 * d, (1, 1))
            self.sep_bn = nn.BatchNorm2d(f1 * d)
            self.pool2 = nn.AvgPool2d((1, 5))
            self.drop2 = nn.Dropout(0.5)

        def forward(self, x):  # (B, 1, C, T) -> (B, F, 1, T')
            x = self.elu(self.t_bn(self.t_conv(x)))
            x = self.elu(self.s_bn(self.spatial_dw(x)))
            x = self.drop1(self.pool1(x))
            x = self.elu(self.sep_bn(self.sep_pw(self.sep_dw(x))))
            return self.drop2(self.pool2(x))

    class TCNBlock(nn.Module):
        """Dilated 2D temporal convolutions with a residual projection."""

        def __init__(self, in_ch: int = 16, out_ch: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, (1, 3),
                                   padding=(0, 2), dilation=(1, 2))
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 3),
                                   padding=(0, 4), dilation=(1, 4))
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.elu = nn.ELU()
            self.proj = nn.Conv2d(in_ch, out_ch, (1, 1))

        def forward(self, x):
            out = self.elu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.elu(out + self.proj(x))

    class EEGTCNet(nn.Module):
        """EEG-TCNet: EEGNet core + TCN -> Softmax."""

        def __init__(self, n_channels: int = N_CHANNELS,
                     n_classes: int = len(TARGET_CLASSES)):
            super().__init__()
            self.n_channels = n_channels
            self.core = EEGNetCore(n_channels)
            self.tcn = TCNBlock(16, 32)
            self.head = nn.Linear(32, n_classes)

        def forward(self, x):  # x: (B, 1, C, T)
            x = self.tcn(self.core(x))       # (B, 32, 1, T')
            return self.head(x.mean(dim=3).squeeze(2))  # mean over time

    class EEGTCNetLSTM(nn.Module):
        """EEG-TCNet + two sequential LSTM layers (30 hidden units each)."""

        def __init__(self, n_channels: int = N_CHANNELS,
                     n_classes: int = len(TARGET_CLASSES)):
            super().__init__()
            self.n_channels = n_channels
            self.core = EEGNetCore(n_channels)
            self.tcn = TCNBlock(16, 32)
            self.lstm = nn.LSTM(32, 30, num_layers=2, batch_first=True)
            self.head = nn.Linear(30, n_classes)

        def forward(self, x):  # x: (B, 1, C, T)
            x = self.tcn(self.core(x)).squeeze(2).permute(0, 2, 1)  # (B, T', 32)
            _, (h, _) = self.lstm(x)
            return self.head(h[-1])


class EEGTCNetClassifier(TorchZooClassifier):
    MODEL_ID = "eeg_tcnet"
    CKPT_NAME = "eeg_tcnet.pth"

    def _build_model(self):
        return EEGTCNet(n_channels=N_CHANNELS, n_classes=len(TARGET_CLASSES))

    def predict(self, eeg: np.ndarray, **_) -> Dict:
        return super().predict(np.asarray(eeg, dtype=np.float64))


class EEGTCNetLSTMClassifier(TorchZooClassifier):
    MODEL_ID = "eeg_tcnet_lstm"
    CKPT_NAME = "eeg_tcnet_lstm.pth"

    def _build_model(self):
        return EEGTCNetLSTM(n_channels=N_CHANNELS,
                            n_classes=len(TARGET_CLASSES))

    def predict(self, eeg: np.ndarray, **_) -> Dict:
        return super().predict(np.asarray(eeg, dtype=np.float64))

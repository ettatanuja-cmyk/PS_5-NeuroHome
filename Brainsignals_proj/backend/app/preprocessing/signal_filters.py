"""Preprocessing pipeline (base paper): Butterworth, Notch, Winsorization.

  1. Resample EEG signals to 120 Hz.
  2. 6th-order Butterworth bandpass filter (1-15 Hz).
  3. Notch filter for power-line noise removal (50 Hz).
  4. Winsorization for outlier handling.
  5. Per-channel z-score normalisation (post-winsorization cleanup).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy import signal as scipy_signal
except Exception:
    scipy_signal = None  # type: ignore

TARGET_FS = 120.0
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 15.0
BUTTER_ORDER = 6
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0
WINSOR_LIMITS = 0.01
DEFAULT_ORIG_FS = 500.0  # dataset recorded at 500 Hz (see param.mat)


def resample_to(eeg: np.ndarray, orig_fs: float,
                target_fs: float = TARGET_FS) -> np.ndarray:
    if orig_fs == target_fs:
        return np.asarray(eeg, dtype=np.float64)
    n_samples = max(int(round(eeg.shape[1] * target_fs / orig_fs)), 8)
    if scipy_signal is not None:
        return scipy_signal.resample(eeg, n_samples, axis=1).astype(np.float64)
    old_idx = np.linspace(0, 1, eeg.shape[1])
    new_idx = np.linspace(0, 1, n_samples)
    return np.stack(
        [np.interp(new_idx, old_idx, eeg[ch]) for ch in range(eeg.shape[0])]
    ).astype(np.float64)


def butter_bandpass(eeg: np.ndarray, fs: float = TARGET_FS,
                    low: float = BANDPASS_LOW, high: float = BANDPASS_HIGH,
                    order: int = BUTTER_ORDER) -> np.ndarray:
    """6th-order Butterworth bandpass filter (1-15 Hz)."""
    x = np.asarray(eeg, dtype=np.float64)
    if scipy_signal is None:
        return x - x.mean(axis=1, keepdims=True)
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order, [low / nyq, high / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if x.shape[1] <= padlen:  # too short for filtfilt -> detrend only
        return x - x.mean(axis=1, keepdims=True)
    return scipy_signal.filtfilt(b, a, x, axis=1).astype(np.float64)


def notch_filter(eeg: np.ndarray, fs: float = TARGET_FS,
                 freq: float = NOTCH_FREQ, q: float = NOTCH_Q) -> np.ndarray:
    """Notch filter for power-line noise removal."""
    x = np.asarray(eeg, dtype=np.float64)
    if scipy_signal is None:
        return x
    try:
        b, a = scipy_signal.iirnotch(freq, q, fs)
        return scipy_signal.filtfilt(b, a, x, axis=1).astype(np.float64)
    except Exception:
        return x


def winsorize(eeg: np.ndarray, limits: float = WINSOR_LIMITS) -> np.ndarray:
    """Winsorization: clip each channel at the limits / 1-limits percentiles."""
    x = np.asarray(eeg, dtype=np.float64)
    out = np.empty_like(x)
    for ch in range(x.shape[0]):
        lo, hi = np.percentile(x[ch], [100 * limits, 100 * (1 - limits)])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[ch] = x[ch]
        else:
            out[ch] = np.clip(x[ch], lo, hi)
    return out


def normalize_outliers(eeg: np.ndarray, clip_std: float = 4.0) -> np.ndarray:
    """Per-channel z-score + clip extreme residuals, then re-standardise."""
    x = np.asarray(eeg, dtype=np.float64)
    out = np.empty_like(x)
    for ch in range(x.shape[0]):
        sd = float(np.std(x[ch]))
        if sd < 1e-9:
            out[ch] = 0.0
            continue
        z = np.clip((x[ch] - float(np.mean(x[ch]))) / sd, -clip_std, clip_std)
        out[ch] = (z - float(np.mean(z))) / (float(np.std(z)) + 1e-9)
    return np.nan_to_num(out, nan=0.0, posinf=4.0, neginf=-4.0)


@dataclass
class PreprocessResult:
    eeg: np.ndarray                    # [channels x time] @ TARGET_FS
    trigger: Optional[np.ndarray]      # raw trigger vector (original rate) or None
    source_key: str
    member_name: str
    orig_shape: Tuple[int, int]
    orig_fs: float
    meta: Dict = field(default_factory=dict)


def preprocess(eeg_raw: np.ndarray, trigger: Optional[np.ndarray] = None,
               orig_fs: float = DEFAULT_ORIG_FS, source_key: str = "",
               member_name: str = "") -> PreprocessResult:
    orig_shape = (int(eeg_raw.shape[0]), int(eeg_raw.shape[1]))
    x = resample_to(eeg_raw, orig_fs, TARGET_FS)
    x = butter_bandpass(x, fs=TARGET_FS)
    x = notch_filter(x, fs=TARGET_FS)
    x = winsorize(x)
    x = normalize_outliers(x)
    return PreprocessResult(
        eeg=x, trigger=trigger, source_key=source_key,
        member_name=member_name, orig_shape=orig_shape, orig_fs=orig_fs,
        meta={
            "target_fs": TARGET_FS,
            "bandpass": [BANDPASS_LOW, BANDPASS_HIGH],
            "butter_order": BUTTER_ORDER,
            "notch_freq": NOTCH_FREQ,
            "winsor_limits": WINSOR_LIMITS,
            "channels": int(x.shape[0]),
            "samples": int(x.shape[1]),
        },
    )


def detect_p300(eeg: np.ndarray,
                trigger: Optional[np.ndarray]) -> Tuple[bool, float]:
    """Heuristic P300 flag: ERP peak 250-500 ms after rare stimulus markers.

    Falls back to a variance/crest-based ERP-likeness score when no trigger
    channel is present (e.g. cal_sig files). Returns (detected, score_0_1).
    """
    try:
        seg = eeg[: min(eeg.shape[0], 8), :]
        if trigger is not None and len(trigger) > 10:
            stim_idx = np.flatnonzero(
                np.isin(np.asarray(trigger).ravel().astype(int), [1, 2, 3, 4, 5, 6]))
            if stim_idx.size > 0:
                ratio = stim_idx.size / max(len(trigger), 1)
                win = seg[:, : min(seg.shape[1], int(TARGET_FS * 2))]
                peak = float(np.max(np.abs(win)))
                score = float(np.clip(
                    0.35 + 0.65 * (1 - min(ratio * 4, 1)) * min(peak / 3.0, 1.0), 0, 1))
                return bool(score > 0.5), score
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        crest = float(np.max(np.abs(seg))) / rms
        score = float(np.clip((crest - 2.0) / 6.0, 0.05, 0.95))
        return bool(score > 0.5), score
    except Exception:
        return False, 0.0

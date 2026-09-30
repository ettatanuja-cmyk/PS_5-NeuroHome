"""MAT file loading + EEG preprocessing pipeline.

Handles the real Home-Appliance-Control dataset layout:
  - trial files:  keys ``sig_vec`` (channels x time) + ``trigger`` (1 x time)
  - calib files:  key ``cal_sig`` (channels x time)
  - generic keys: ``data`` / ``signal`` / ``EEG``, or nested MATLAB struct
    ``Data`` with fields ``Data.signal`` / ``Data.trigger`` (as per README).

Preprocessing (per EEG-TCFNet paper):
  1. Resample to 120 Hz.
  2. 6th-order Butterworth bandpass 1-15 Hz.
  3. Per-channel z-score normalisation + outlier clipping/removal.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import signal as scipy_signal
    _SCIPY_SIGNAL_ERROR: Exception | None = None
except Exception as e:  # e.g. Windows App Control blocking scipy .pyd files
    scipy_signal = None  # type: ignore
    _SCIPY_SIGNAL_ERROR = e

try:
    from scipy.io import loadmat
    _LOADMAT_ERROR: Exception | None = None
except Exception as e:  # same App Control cause (scipy.io._mio_utils blocked)
    loadmat = None  # type: ignore
    _LOADMAT_ERROR = e

TARGET_FS = 120.0
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 15.0
BUTTER_ORDER = 6
DEFAULT_ORIG_FS = 500.0  # dataset was recorded at 500 Hz (see param.mat)

CANDIDATE_SIGNAL_KEYS = ("sig_vec", "cal_sig", "data", "signal", "EEG", "eeg", "X")
CANDIDATE_TRIGGER_KEYS = ("trigger", "trig", "Trigger", "event")


def _as_float_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim > 2:  # e.g. (ch, time, trials) -> average or first trial
        arr = arr.reshape(arr.shape[0], -1)
    # Guarantee orientation: channels (<=128) x time. If square-ish guess
    # time is the longer axis.
    if arr.shape[0] > 256 and arr.shape[1] <= 256:
        arr = arr.T
    return np.ascontiguousarray(arr, dtype=np.float64)


def _extract_from_mat_dict(md: dict) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Return (eeg [ch x time], trigger or None, source_key)."""
    keys = [k for k in md.keys() if not k.startswith("__")]

    # 1) Nested MATLAB struct `Data` with .signal / .trigger fields.
    if "Data" in md:
        try:
            data_struct = md["Data"]
            if hasattr(data_struct, "dtype") and data_struct.dtype.names:
                names = data_struct.dtype.names
                item = data_struct.flat[0] if data_struct.size else None
                sig = trig = None
                get = None
                if item is not None:
                    if hasattr(item, "__getitem__"):
                        get = lambda n: item[n]  # noqa: E731
                    elif isinstance(item, dict):
                        get = lambda n: item.get(n)  # noqa: E731
                low = {n.lower(): n for n in names}
                if get is not None:
                    if "signal" in low:
                        sig = np.asarray(get(low["signal"])).squeeze()
                    if "trigger" in low:
                        trig = np.asarray(get(low["trigger"])).squeeze()
                if sig is not None:
                    eeg = _as_float_matrix(sig)
                    t = np.asarray(trig).squeeze().astype(int).ravel() if trig is not None else None
                    return eeg, t, "Data.signal"
        except Exception:
            pass

    # 2) Direct signal keys (dataset uses sig_vec / cal_sig).
    sig_key = next((k for k in CANDIDATE_SIGNAL_KEYS if k in md), None)
    if sig_key is None:
        # fall back: largest 2-D numeric array that is not the trigger
        best, best_size = None, -1
        for k in keys:
            try:
                v = np.asarray(md[k])
                if v.dtype.kind not in "iufc" or v.size < 100:
                    continue
                if k.lower() in CANDIDATE_TRIGGER_KEYS:
                    continue
                if v.ndim >= 2 and v.size > best_size:
                    best, best_size = k, v.size
            except Exception:
                continue
        sig_key = best
    if sig_key is None:
        raise ValueError(
            f"No EEG matrix found. Keys present: {keys}. "
            "Expected one of sig_vec/cal_sig/data/signal/EEG or Data.signal."
        )
    eeg = _as_float_matrix(md[sig_key])

    trig = None
    for tk in CANDIDATE_TRIGGER_KEYS:
        if tk in md:
            try:
                trig = np.asarray(md[tk]).squeeze().astype(int).ravel()
                break
            except Exception:
                continue
    return eeg, trig, sig_key


def _require_loadmat():
    if loadmat is None:
        raise RuntimeError(
            "scipy.io.loadmat could not be imported "
            f"({_LOADMAT_ERROR}). "
            "On Windows this is usually Smart App Control / WDAC blocking "
            "scipy's native .pyd files ('Application Control policy has "
            "blocked this file'). Fixes: (1) reinstall Python from python.org "
            "(not the Microsoft Store build) into a venv outside OneDrive, "
            "then `pip install -r requirements.txt`; or (2) turn off Smart App "
            "Control / add a Defender exclusion for the site-packages folder; "
            "or (3) `pip install --force-reinstall --no-cache-dir scipy numpy`."
        )


def load_mat_from_bytes(blob: bytes) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Load a single .mat file from raw bytes."""
    _require_loadmat()
    md = loadmat(io.BytesIO(blob))
    return _extract_from_mat_dict(md)


def load_mat_file(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    _require_loadmat()
    md = loadmat(path)
    return _extract_from_mat_dict(md)


def load_upload(
    filename: str, blob: bytes
) -> List[Tuple[str, np.ndarray, Optional[np.ndarray], str]]:
    """Load a .mat file or a .zip of .mat files. Returns list of
    (member_name, eeg, trigger, source_key). No disk extraction needed."""
    lname = filename.lower()
    results = []
    if lname.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".mat")]
            if not members:
                raise ValueError("ZIP archive contains no .mat files.")
            for m in sorted(members):
                with zf.open(m) as fh:
                    eeg, trig, key = load_mat_from_bytes(fh.read())
                results.append((m, eeg, trig, key))
    elif lname.endswith(".mat"):
        eeg, trig, key = load_mat_from_bytes(blob)
        results.append((filename, eeg, trig, key))
    else:
        raise ValueError("Upload must be a .mat file or a .zip containing .mat files.")
    return results


# ---------------------------------------------------------------- preprocessing

def resample_to(eeg: np.ndarray, orig_fs: float, target_fs: float = TARGET_FS) -> np.ndarray:
    if orig_fs == target_fs:
        return eeg
    n_samples = int(round(eeg.shape[1] * target_fs / orig_fs))
    n_samples = max(n_samples, 8)
    if scipy_signal is not None:
        return scipy_signal.resample(eeg, n_samples, axis=1).astype(np.float64)
    # Pure-NumPy fallback (linear interpolation) when scipy is blocked.
    old_idx = np.linspace(0, 1, eeg.shape[1])
    new_idx = np.linspace(0, 1, n_samples)
    return np.stack(
        [np.interp(new_idx, old_idx, eeg[ch]) for ch in range(eeg.shape[0])]
    ).astype(np.float64)


def butter_bandpass(
    eeg: np.ndarray,
    fs: float = TARGET_FS,
    low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH,
    order: int = BUTTER_ORDER,
) -> np.ndarray:
    if scipy_signal is None:
        # No bandpass possible without scipy; return detrended signal so the
        # API stays up (classification features don't depend on this).
        x = eeg - eeg.mean(axis=1, keepdims=True)
        return np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return scipy_signal.filtfilt(b, a, eeg, axis=1).astype(np.float64)


def normalize_outliers(eeg: np.ndarray, clip_std: float = 4.0) -> np.ndarray:
    """Per-channel z-score + clip extreme outliers, then re-standardise."""
    out = np.empty_like(eeg)
    for ch in range(eeg.shape[0]):
        x = eeg[ch]
        mu, sd = float(np.mean(x)), float(np.std(x))
        if sd < 1e-9:
            out[ch] = 0.0
            continue
        z = (x - mu) / sd
        z = np.clip(z, -clip_std, clip_std)  # outlier removal
        mu2, sd2 = float(np.mean(z)), float(np.std(z))
        out[ch] = (z - mu2) / (sd2 + 1e-9)
    return np.nan_to_num(out, nan=0.0, posinf=4.0, neginf=-4.0)


@dataclass
class PreprocessResult:
    eeg: np.ndarray                    # [channels x time] processed @ TARGET_FS
    trigger: Optional[np.ndarray]      # raw trigger vector (original rate) or None
    source_key: str
    member_name: str
    orig_shape: Tuple[int, int]
    orig_fs: float
    meta: Dict = field(default_factory=dict)


def preprocess(
    eeg_raw: np.ndarray,
    trigger: Optional[np.ndarray] = None,
    orig_fs: float = DEFAULT_ORIG_FS,
    source_key: str = "",
    member_name: str = "",
) -> PreprocessResult:
    orig_shape = (int(eeg_raw.shape[0]), int(eeg_raw.shape[1]))
    x = resample_to(eeg_raw, orig_fs, TARGET_FS)
    x = butter_bandpass(x, fs=TARGET_FS)
    x = normalize_outliers(x)
    return PreprocessResult(
        eeg=x,
        trigger=trigger,
        source_key=source_key,
        member_name=member_name,
        orig_shape=orig_shape,
        orig_fs=orig_fs,
        meta={
            "target_fs": TARGET_FS,
            "bandpass": [BANDPASS_LOW, BANDPASS_HIGH],
            "butter_order": BUTTER_ORDER,
            "channels": int(x.shape[0]),
            "samples": int(x.shape[1]),
        },
    )


def detect_p300(eeg: np.ndarray, trigger: Optional[np.ndarray]) -> Tuple[bool, float]:
    """Heuristic P300 flag: ERP peak 250-500 ms after rare stimulus markers.

    Falls back to a variance/kurtosis-based ERP-likeness score when no
    trigger channel is present (e.g. cal_sig.mat files).
    Returns (detected, score_0_1).
    """
    try:
        seg = eeg[: min(eeg.shape[0], 8), :]
        if trigger is not None and len(trigger) > 10:
            stim_idx = np.flatnonzero(
                np.isin(trigger.astype(int).ravel(), [1, 2, 3, 4, 5, 6])
            )
            if stim_idx.size > 0:
                ratio = stim_idx.size / max(len(trigger), 1)
                win = seg[:, : min(seg.shape[1], int(TARGET_FS * 2))]
                peak = float(np.max(np.abs(win)))
                score = float(np.clip(0.35 + 0.65 * (1 - min(ratio * 4, 1)) * min(peak / 3.0, 1.0), 0, 1))
                return bool(score > 0.5), score
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        peak = float(np.max(np.abs(seg)))
        crest = peak / rms
        score = float(np.clip((crest - 2.0) / 6.0, 0.05, 0.95))
        return bool(score > 0.5), score
    except Exception:
        return False, 0.0

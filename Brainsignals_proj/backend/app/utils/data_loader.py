"""Loader for https://github.com/jml226/Home-Appliance-Control-Dataset.

Trial files: ``sig_vec`` (channels x time) + ``trigger`` (1 x time).
Calibration files: ``cal_sig`` (channels x time, no trigger).
Generic files: ``data`` / ``signal`` / ``EEG``, nested MATLAB struct
``Data`` (.signal / .trigger), plain ``.csv`` (rows = time samples,
optional ``trigger`` column) and ``.npy`` / ``.npz`` arrays.

Dataset root: ``Home-Appliance-Control-Dataset-main/`` at the project root
(<Appliance>/<subject>/*.mat plus param.mat with Fs=500).
"""

from __future__ import annotations

import glob
import io
import os
import zipfile
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.io import loadmat
    _LOADMAT_ERROR: Exception | None = None
except Exception as e:
    loadmat = None  # type: ignore
    _LOADMAT_ERROR = e

DEFAULT_ORIG_FS = 500.0

CANDIDATE_SIGNAL_KEYS = ("sig_vec", "cal_sig", "data", "signal", "EEG", "eeg", "X")
CANDIDATE_TRIGGER_KEYS = ("trigger", "trig", "Trigger", "event")

APPLIANCE_FOLDERS = {
    "Air Conditioner": "AirConditioner",
    "Bluetooth Speaker": "BluetoothSpeaker",
    "Door Lock": "Doorlock",
    "Electric Light": "ElectricLight",
    "TV": "TV",
}


def _as_float_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] > 256 and arr.shape[1] <= 256:
        arr = arr.T
    return np.ascontiguousarray(arr, dtype=np.float64)


def _extract_from_mat_dict(md: dict) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    keys = [k for k in md.keys() if not k.startswith("__")]
    if "Data" in md:  # nested MATLAB struct with .signal / .trigger
        try:
            data_struct = md["Data"]
            if hasattr(data_struct, "dtype") and data_struct.dtype.names:
                names = data_struct.dtype.names
                item = data_struct.flat[0] if data_struct.size else None
                get = None
                if item is not None:
                    if hasattr(item, "__getitem__"):
                        get = lambda n: item[n]  # noqa: E731
                    elif isinstance(item, dict):
                        get = lambda n: item.get(n)  # noqa: E731
                low = {n.lower(): n for n in names}
                sig = trig = None
                if get is not None:
                    if "signal" in low:
                        sig = np.asarray(get(low["signal"])).squeeze()
                    if "trigger" in low:
                        trig = np.asarray(get(low["trigger"])).squeeze()
                if sig is not None:
                    eeg = _as_float_matrix(sig)
                    t = (np.asarray(trig).squeeze().astype(int).ravel()
                         if trig is not None else None)
                    return eeg, t, "Data.signal"
        except Exception:
            pass
    sig_key = next((k for k in CANDIDATE_SIGNAL_KEYS if k in md), None)
    if sig_key is None:
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
            "Expected one of sig_vec/cal_sig/data/signal/EEG or Data.signal.")
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


def _require_loadmat() -> None:
    if loadmat is None:
        raise RuntimeError(
            "scipy.io.loadmat could not be imported "
            f"({_LOADMAT_ERROR}). "
            "On Windows this is usually Smart App Control / WDAC blocking "
            "scipy's native .pyd files. Fix: install Python from python.org "
            "(not the Microsoft Store build) into a venv outside OneDrive, "
            "then `pip install -r requirements.txt`.")


def load_mat_from_bytes(blob: bytes) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    _require_loadmat()
    return _extract_from_mat_dict(loadmat(io.BytesIO(blob)))


def load_mat_file(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    _require_loadmat()
    return _extract_from_mat_dict(loadmat(path))


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_csv_from_bytes(blob: bytes) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """CSV sample file: rows = time samples, cols = channels.

    An optional ``trigger`` column is split off when present.
    """
    text = blob.decode("utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty CSV file.")
    header = [h.strip() for h in lines[0].split(",")]
    has_header = any(not _is_float(h) for h in header)
    rows = []
    for ln in lines[1:] if has_header else lines:
        try:
            rows.append([float(v) for v in ln.split(",") if v.strip() != ""])
        except ValueError:
            continue
    if not rows:
        raise ValueError("No numeric data found in CSV.")
    arr = np.asarray(rows, dtype=np.float64)
    trig = None
    if has_header and "trigger" in [h.lower() for h in header]:
        ti = [h.lower() for h in header].index("trigger")
        if arr.shape[1] > ti:
            trig = arr[:, ti].astype(int).ravel()
            arr = np.delete(arr, ti, axis=1)
    eeg = np.ascontiguousarray(arr.T, dtype=np.float64)  # samples x ch -> ch x time
    return eeg, trig, "csv(trigger-col)" if trig is not None else "csv"


def load_npy_from_bytes(blob: bytes) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """``.npy`` array (channels x time) or ``.npz`` dict (data + trigger)."""
    arr = np.load(io.BytesIO(blob), allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        keys = list(arr.keys())
        sig_key = next((k for k in ("sig_vec", "cal_sig", "data", "signal",
                                    "eeg", "X") if k in arr), keys[0])
        eeg = _as_float_matrix(np.asarray(arr[sig_key]))
        trig = None
        for tk in ("trigger", "trig"):
            if tk in arr:
                try:
                    trig = np.asarray(arr[tk]).squeeze().astype(int).ravel()
                except Exception:
                    trig = None
        return eeg, trig, f"npz:{sig_key}"
    return _as_float_matrix(np.asarray(arr)), None, "npy"


def load_upload(filename: str, blob: bytes
                ) -> List[Tuple[str, np.ndarray, Optional[np.ndarray], str]]:
    """Load a .mat / .csv / .npy sample file or a .zip of those."""
    lname = filename.lower()
    if lname.endswith(".zip"):
        results = []
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            members = [m for m in zf.namelist()
                       if m.lower().endswith((".mat", ".csv", ".npy", ".npz"))]
            if not members:
                raise ValueError("ZIP archive contains no .mat/.csv/.npy files.")
            for m in sorted(members):
                with zf.open(m) as fh:
                    raw = fh.read()
                ml = m.lower()
                if ml.endswith(".mat"):
                    eeg, trig, key = load_mat_from_bytes(raw)
                elif ml.endswith(".csv"):
                    eeg, trig, key = load_csv_from_bytes(raw)
                else:
                    eeg, trig, key = load_npy_from_bytes(raw)
                results.append((m, eeg, trig, key))
        return results
    if lname.endswith(".mat"):
        eeg, trig, key = load_mat_from_bytes(blob)
    elif lname.endswith(".csv"):
        eeg, trig, key = load_csv_from_bytes(blob)
    elif lname.endswith((".npy", ".npz")):
        eeg, trig, key = load_npy_from_bytes(blob)
    else:
        raise ValueError("Upload must be .mat, .csv, .npy or a .zip of those.")
    return [(filename, eeg, trig, key)]


def find_dataset_root(start: str = "") -> str:
    """Locate ``Home-Appliance-Control-Dataset-main`` walking up from start."""
    base = start or os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        cand = os.path.join(base, "Home-Appliance-Control-Dataset-main")
        if os.path.isdir(cand):
            return cand
        base = os.path.dirname(base)
    return os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "Home-Appliance-Control-Dataset-main"))


def iter_trial_files(root: str = "", pattern: str = "*.mat"):
    """Yield ``(appliance_name, path)`` for every trial .mat in the dataset."""
    root = root or find_dataset_root()
    for app_name, folder in APPLIANCE_FOLDERS.items():
        for path in sorted(glob.glob(os.path.join(root, folder, "*", pattern))):
            if "param" in os.path.basename(path).lower():
                continue
            yield app_name, path

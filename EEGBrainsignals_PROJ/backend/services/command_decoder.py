"""Target-command decoder: which stimulus (command) was the user attending?

Method (standard ERP oddball decoding, no labels required):
  1. For every stimulus marker (1-6) in the trigger channel, cut an epoch
     (0-600 ms post-stimulus) out of the preprocessed 120 Hz EEG.
  2. Baseline-correct each epoch, average across repetitions/channels to get
     one ERP waveform per stimulus type.
  3. Score = mean amplitude in the P300 window (250-500 ms). The stimulus
     with the largest P300 is the decoded target -> mapped to the
     appliance's command label (stimulus_id -> command).

Returns None when no trigger channel exists (e.g. cal_sig.mat files).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

FS = 120.0
EPOCH_MS = 600
P300_START_MS = 250
P300_END_MS = 500
BASELINE_MS = 100


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def decode_command(
    eeg: np.ndarray,
    trigger: Optional[np.ndarray],
    orig_fs: float,
    commands: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Decode attended stimulus id + per-command probabilities."""
    if trigger is None or len(trigger) < 10 or not commands:
        return None
    try:
        trig = np.asarray(trigger).ravel()
        scale = FS / float(orig_fs)  # trigger idx (orig rate) -> eeg idx (120 Hz)
        n_epoch = int(EPOCH_MS * FS / 1000)          # 72 samples
        w0, w1 = int(P300_START_MS * FS / 1000), int(P300_END_MS * FS / 1000)
        b0 = int(BASELINE_MS * FS / 1000)
        t_len = eeg.shape[1]

        valid_ids = [c["stimulus_id"] for c in commands]
        erps: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        for sid in valid_ids:
            idx = np.flatnonzero(trig == sid)
            waves = []
            for j in idx:
                s = int(round(j * scale))
                if s < 0 or s + n_epoch > t_len:
                    continue
                ep = eeg[:, s:s + n_epoch].mean(axis=0)  # avg channels
                # pre-stimulus baseline when available, else epoch mean
                if s - b0 >= 0:
                    ep = ep - eeg[:, s - b0:s].mean()
                else:
                    ep = ep - ep[:b0].mean()
                waves.append(ep)
            if waves:
                erps[sid] = np.mean(np.stack(waves), axis=0)
                counts[sid] = len(waves)
        if len(erps) < 2:
            return None

        scores = {sid: float(erps[sid][w0:w1].mean()) for sid in erps}
        ids = sorted(scores)
        logits = np.array([scores[i] for i in ids])
        # standardise before softmax so probabilities reflect separation
        sd = float(logits.std())
        probs = _softmax(logits / (sd + 1e-9) * 1.5)
        best = ids[int(np.argmax(probs))]
        label = next(c["label"] for c in commands if c["stimulus_id"] == best)
        return {
            "stimulus_id": int(best),
            "label": label,
            "confidence": round(float(probs.max()), 4),
            "command_scores": {
                next(c["label"] for c in commands if c["stimulus_id"] == i): round(float(p), 4)
                for i, p in zip(ids, probs)
            },
            "erp_scores": {str(i): round(float(scores[i]), 4) for i in ids},
            "repetitions": {str(i): counts[i] for i in ids},
            "method": "P300 window (250-500 ms) mean amplitude, 0-600 ms epochs @120Hz",
        }
    except Exception:
        return None

"""Train the base-paper model zoo on the Home-Appliance-Control dataset.

Trains, in one run:
  * traditional ML baselines (Logistic Regression, SVM-RBF, MLP) on
    window-feature vectors  -> ``app/models/weights/traditional_ml.pkl``
  * LeNet, EEG-TCNet, EEG-TCNet+LSTM, EEG-TCFNet on 31-ch x 12 s windows
    (4 fixed + 1 fresh random window per file per epoch)
    -> ``app/models/weights/{lenet,eeg_tcnet,eeg_tcnet_lstm,eeg_tcfnet}.pth``

All checkpoints are auto-loaded by the API. Subject-grouped held-out split.

Run from inside backend/:
    python train.py [--max-per-class 80] [--epochs 20]
Requires: torch, scikit-learn (see requirements.txt).
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.common import (  # noqa: E402
    N_CHANNELS,
    TARGET_CLASSES,
    WEIGHTS_DIR,
    WIN_SAMPLES,
    _fit_channels,
    _windows,
)
from app.models.traditional_ml import (  # noqa: E402
    extract_window_features,
    fit_mlp,
    fit_rff_svm,
    fit_softmax,
    standardize_fit,
)
from app.preprocessing.signal_filters import preprocess  # noqa: E402
from app.utils.data_loader import iter_trial_files, load_mat_file  # noqa: E402

CAL_SKIP = ("cal_sig", "param")


def preload(max_per_class: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for app_name in TARGET_CLASSES:
        files = [p for a, p in iter_trial_files() if a == app_name]
        files = [p for p in files
                 if all(s not in os.path.basename(p).lower() for s in CAL_SKIP)]
        rng.shuffle(files)
        for f in files[:max_per_class]:
            group = f"{app_name}/{os.path.basename(os.path.dirname(f))}"
            rows.append((app_name, f, group))
    print(f"files: {len(rows)}", flush=True)
    arrays, labels, groups = [], [], []
    for app_name, path, group in rows:
        try:
            eeg_raw, trig, _ = load_mat_file(path)
            prep = preprocess(eeg_raw, trig, orig_fs=500.0)
            seg = _fit_channels(prep.eeg, N_CHANNELS)
            if seg.shape[1] < 64:
                print(f"skip {path}: too short")
                continue
            arrays.append(seg.astype(np.float32))
            labels.append(TARGET_CLASSES.index(app_name))
            groups.append(group)
        except Exception as e:
            print(f"skip {path}: {e}")
    print(f"loaded: {len(arrays)}", flush=True)
    return arrays, np.array(labels), np.array(groups)


def fixed_windows(seg: np.ndarray, n: int = 4):
    t = seg.shape[1]
    if t <= WIN_SAMPLES:
        w = np.pad(seg, ((0, 0), (0, WIN_SAMPLES - t)))
        return [w[np.newaxis, :, :] for _ in range(n)]
    starts = np.linspace(0, t - WIN_SAMPLES, n).round().astype(int).tolist()
    return [seg[:, s:s + WIN_SAMPLES][np.newaxis, :, :] for s in starts]


def random_window(seg: np.ndarray, rng):
    t = seg.shape[1]
    if t <= WIN_SAMPLES:
        w = np.pad(seg, ((0, 0), (0, WIN_SAMPLES - t)))
    else:
        s = int(rng.integers(0, t - WIN_SAMPLES + 1))
        w = seg[:, s:s + WIN_SAMPLES]
    return w[np.newaxis, :, :]


def seed_fnb(model, xb):
    """Fuzzy c-means seeding of a TCFNet model's FNB centroids."""
    import torch as _torch  # noqa: F401

    with _torch.no_grad():
        t = model.elu(model.bn1(model.conv_temporal(xb)))
        t = model.elu(model.bn2(model.conv_depthwise(t)))
        t = model.drop1(model.pool1(t))
        t = model.elu(model.bn3(model.conv_separable_pw(
            model.conv_separable_dw(t))))
        t = model.drop2(model.pool2(t))
        t = model.elu(model.tcn_bn(model.tcn(t)))
        t = t.squeeze(2).permute(0, 2, 1)
        _, (h, _) = model.lstm(t)
        model.fnb.init_centers_fcm(h[-1])


def train_torch_model(name, builder, arrays, y, tr_idx, te_idx, args, rng,
                      seed_fnb_fn=None):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n=== {name} ===", flush=True)
    torch.manual_seed(args.seed)
    model = builder()
    if seed_fnb_fn is not None:
        with torch.no_grad():
            xb = torch.tensor(np.stack(
                [random_window(arrays[i], rng) for i in tr_idx[:args.batch]]))
        seed_fnb_fn(model, xb)
        print("FNB centroids seeded with fuzzy c-means", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def multi_window_predict(idxs):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in idxs:
                wins = _windows(arrays[i].astype(np.float64))
                batch = torch.tensor(
                    np.stack([w[np.newaxis, :, :] for w in wins]),
                    dtype=torch.float32)
                logits = model(batch).cpu().numpy().mean(axis=0)
                preds.append(int(np.argmax(logits)))
        return np.array(preds)

    for ep in range(1, args.epochs + 1):
        model.train()
        X = np.concatenate([
            np.stack([w for i in tr_idx for w in fixed_windows(arrays[i])]),
            np.stack([random_window(arrays[i], rng) for i in tr_idx]),
        ])
        y_ep = np.concatenate([np.repeat(y[tr_idx], 4), y[tr_idx]])
        perm = rng.permutation(len(X))
        dl = DataLoader(TensorDataset(torch.tensor(X[perm]),
                                      torch.tensor(y_ep[perm])),
                        batch_size=args.batch, shuffle=True)
        tot, n = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(xb)
            n += len(xb)
        acc_tr = float((multi_window_predict(tr_idx) == y[tr_idx]).mean())
        acc_te = float((multi_window_predict(te_idx) == y[te_idx]).mean())
        print(f"epoch {ep}/{args.epochs} loss={tot / max(n, 1):.4f} "
              f"train={acc_tr:.3f} held-out={acc_te:.3f}", flush=True)

    pred_te = multi_window_predict(te_idx)
    for i, c in enumerate(TARGET_CLASSES):
        mask = y[te_idx] == i
        rec = float((pred_te[mask] == i).mean()) if mask.sum() else float("nan")
        print(f"recall {c:18s}: {rec:.3f} (n={int(mask.sum())})")
    return model


def train_sklearn(arrays, y, tr_idx, te_idx, seed):
    print("\n=== traditional ML (logreg / svm / mlp, pure NumPy) ===", flush=True)
    Xtr = np.stack([f for i in tr_idx
                    for f in [extract_window_features(w)
                              for w in _windows(arrays[i].astype(np.float64))]])
    ytr = np.repeat(y[tr_idx], [len(_windows(arrays[i])) for i in tr_idx])
    Xte = np.stack([f for i in te_idx
                    for f in [extract_window_features(w)
                              for w in _windows(arrays[i].astype(np.float64))]])
    yte = np.repeat(y[te_idx], [len(_windows(arrays[i])) for i in te_idx])
    print(f"window sets: train {Xtr.shape}, held-out {Xte.shape}", flush=True)
    scaler = standardize_fit(Xtr)
    from app.models.traditional_ml import (  # noqa: E402
        _forward_proba,
        standardize_apply,
    )

    def scaler_apply(X):
        return standardize_apply(scaler, X)

    fitters = {
        "logreg": lambda: fit_softmax(scaler_apply(Xtr), ytr,
                                      len(TARGET_CLASSES), seed=seed),
        "svm": lambda: fit_rff_svm(scaler_apply(Xtr), ytr,
                                   len(TARGET_CLASSES), seed=seed),
        "mlp": lambda: fit_mlp(scaler_apply(Xtr), ytr,
                               len(TARGET_CLASSES), seed=seed),
    }

    models = {}
    for key, fit in fitters.items():
        models[key] = fit()
        P = _forward_proba(models[key], scaler_apply(Xte))
        acc = float((P.argmax(axis=1) == yte).mean())
        print(f"{key:8s} window-level held-out accuracy: {acc:.3f}", flush=True)
    # file-level accuracy via mean-proba voting
    for key in models:
        ok, n = 0, 0
        for i in te_idx:
            feats = scaler_apply(np.stack(
                [extract_window_features(w) for w in
                 _windows(arrays[i].astype(np.float64))]))
            pred = int(np.argmax(_forward_proba(models[key], feats).mean(axis=0)))
            ok += pred == int(y[i])
            n += 1
        print(f"{key:8s} file-level held-out accuracy: {ok / max(n, 1):.3f}",
              flush=True)
    return {"models": models, "scaler": scaler,
            "classes": TARGET_CLASSES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=str, default="",
                    help="comma list subset: eeg_tcfnet,eeg_tcnet_lstm,"
                         "eeg_tcnet,lenet,traditional_ml (default: all)")
    args = ap.parse_args()
    only = set(s.strip() for s in args.only.split(",") if s.strip())

    arrays, y, groups = preload(args.max_per_class, args.seed)
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    te_groups = set(uniq[: max(1, int(round(len(uniq) * 0.25)))])
    te = np.array([g in te_groups for g in groups])
    tr = ~te
    tr_idx = np.flatnonzero(tr)
    te_idx = np.flatnonzero(te)
    print(f"train files: {tr.sum()}, held-out files: {te.sum()}", flush=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    def want(name):
        return not only or name in only

    if want("traditional_ml"):
        bundle = train_sklearn(arrays, y, tr_idx, te_idx, args.seed)
        with open(os.path.join(WEIGHTS_DIR, "traditional_ml.pkl"), "wb") as fh:
            pickle.dump(bundle, fh)
        print("saved traditional_ml.pkl")

    torch_configs = []
    if want("eeg_tcfnet"):
        from app.models.eeg_tcfnet import EEGTCFNet

        torch_configs.append(("eeg_tcfnet",
                              lambda: EEGTCFNet(n_channels=N_CHANNELS,
                                                n_classes=len(TARGET_CLASSES)),
                              seed_fnb))
    if want("eeg_tcnet_lstm"):
        from app.models.eeg_tcnet import EEGTCNetLSTM

        torch_configs.append(("eeg_tcnet_lstm",
                              lambda: EEGTCNetLSTM(n_channels=N_CHANNELS,
                                                   n_classes=len(TARGET_CLASSES)),
                              None))
    if want("eeg_tcnet"):
        from app.models.eeg_tcnet import EEGTCNet

        torch_configs.append(("eeg_tcnet",
                              lambda: EEGTCNet(n_channels=N_CHANNELS,
                                               n_classes=len(TARGET_CLASSES)),
                              None))
    if want("lenet"):
        from app.models.lenet import LeNet

        torch_configs.append(("lenet",
                              lambda: LeNet(n_classes=len(TARGET_CLASSES)),
                              None))

    if torch_configs:
        import torch

        for name, builder, seed_fn in torch_configs:
            model = train_torch_model(name, builder, arrays, y, tr_idx,
                                      te_idx, args, rng, seed_fn)
            torch.save({"model": model.state_dict(),
                        "classes": TARGET_CLASSES,
                        "n_channels": N_CHANNELS,
                        "win": WIN_SAMPLES},
                       os.path.join(WEIGHTS_DIR, f"{name}.pth"))
            print(f"saved {name}.pth")


if __name__ == "__main__":
    main()

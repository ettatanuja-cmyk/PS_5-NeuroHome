"""Train the appliance-level classifier on the real dataset.

Walks Home-Appliance-Control-Dataset-main/<Appliance>/*/*.mat, extracts the
shared 16 features per file, trains multinomial LogisticRegression with a
subject-grouped split, prints accuracy + confusion matrix, and saves weights
to models/appliance_clf.npz (pure NumPy checkpoint, no pickle needed).

Run from inside backend/:
    python train_appliance_clf.py [--max-per-class 80]
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.appliance_features import extract_features, FEATURE_NAMES  # noqa: E402
from services.data_loader import load_mat_file  # noqa: E402

APPLIANCE_FOLDERS = {
    "Air Conditioner": "AirConditioner",
    "Bluetooth Speaker": "BluetoothSpeaker",
    "Door Lock": "Doorlock",
    "Electric Light": "ElectricLight",
    "TV": "TV",
}
CLASSES = list(APPLIANCE_FOLDERS.keys())
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "Home-Appliance-Control-Dataset-main")


def collect(max_per_class: int, seed: int = 0):
    rng = random.Random(seed)
    rows = []  # (appliance, path, group)
    for app, folder in APPLIANCE_FOLDERS.items():
        files = sorted(glob.glob(os.path.join(DATASET_ROOT, folder, "*", "*.mat")))
        files = [f for f in files if "param" not in os.path.basename(f).lower()]
        rng.shuffle(files)
        for f in files[:max_per_class]:
            group = f"{app}/{os.path.basename(os.path.dirname(f))}"
            rows.append((app, f, group))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()

    try:
        from scipy.optimize import minimize
    except Exception as e:
        print(
            "ERROR: scipy.optimize could not be imported.\n"
            f"  {type(e).__name__}: {e}\n"
            "On Windows this is usually Smart App Control / WDAC blocking "
            "scipy's native .pyd files ('Application Control policy has "
            "blocked this file', e.g. _lsap, _mio_utils, pyduccfft).\n"
            "Fix: reinstall Python from python.org (NOT the Microsoft Store "
            "build), create a fresh venv, then:\n"
            "  pip install --no-cache-dir -r requirements.txt\n"
            "Or turn off Smart App Control / add a Defender exclusion for "
            "the site-packages folder."
        )
        raise SystemExit(1)

    rows = collect(args.max_per_class, args.seed)
    print(f"files: {len(rows)}", flush=True)
    X, y, groups = [], [], []
    for app, path, group in rows:
        try:
            eeg, trig, _ = load_mat_file(path)
            X.append(extract_features(eeg, trig))
            y.append(CLASSES.index(app))
            groups.append(group)
        except Exception as e:
            print(f"skip {path}: {e}")
    X = np.stack(X)
    y = np.array(y)
    groups = np.array(groups)
    print(f"feature matrix: {X.shape}", flush=True)

    # ---- grouped split: whole subject folders go to train OR test ----
    rng = np.random.default_rng(args.seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_te = max(1, int(round(len(uniq) * 0.25)))
    te_groups = set(uniq[:n_te])
    te = np.array([g in te_groups for g in groups])
    tr = ~te

    # ---- manual standardise ----
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0) + 1e-9
    Xs = (X - mu) / sd

    # ---- multinomial logistic regression via L-BFGS (pure NumPy+SciPy) ----
    K = len(CLASSES)
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0

    def pack(W, b):
        return np.concatenate([W.ravel(), b])

    def loss_grad(theta):
        W = theta[: X.shape[1] * K].reshape(X.shape[1], K)
        b = theta[X.shape[1] * K:]
        Z = Xs[tr] @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        n = int(tr.sum())
        loss = float(-np.sum(Y[tr] * np.log(P + 1e-12)) / n
                     + 0.5 * args.l2 * float(np.sum(W * W)) / n)
        G = Xs[tr].T @ (P - Y[tr]) / n + args.l2 * W / n
        Gb = (P - Y[tr]).sum(axis=0) / n
        return loss, pack(G, Gb)

    theta0 = np.zeros(X.shape[1] * K + K)
    res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000})
    W = res.x[: X.shape[1] * K].reshape(X.shape[1], K)
    b = res.x[X.shape[1] * K:]
    print(f"final loss: {res.fun:.4f} ({res.nit} iters)", flush=True)

    def predict_proba(Xq):
        Z = (Xq - mu) / sd @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    for name, idx in (("train", tr), ("held-out subjects", te)):
        pred = predict_proba(X[idx]).argmax(axis=1)
        acc = float((pred == y[idx]).mean())
        print(f"{name} accuracy: {acc:.3f} (n={int(idx.sum())})")
    pred = predict_proba(X[te]).argmax(axis=1)
    print("confusion rows=true, cols=pred:")
    print(" " * 18 + "".join(f"{c[:6]:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        row = [(pred[y[te] == i] == j).sum() for j in range(K)]
        print(f"{c:18s}" + "".join(f"{v:>8d}" for v in row))
    for i, c in enumerate(CLASSES):
        mask = y[te] == i
        rec = float((pred[mask] == i).mean()) if mask.sum() else float("nan")
        print(f"recall {c:18s}: {rec:.3f} (n={int(mask.sum())})")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "models", "appliance_clf.npz")
    # store transposed coef (K x F) to match sklearn convention
    np.savez(out, classes=np.array(CLASSES), coef_=W.T,
             intercept_=b, mean_=mu, scale_=sd,
             features=np.array(FEATURE_NAMES))
    print(f"saved {out}")


if __name__ == "__main__":
    main()

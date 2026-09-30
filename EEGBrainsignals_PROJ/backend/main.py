"""FastAPI backend: multi-model BCI home-appliance prediction service.

Supports dynamic model selection via ``model_name`` on ``POST /api/predict``:

  - ``eeg_tcfnet`` (default): EEGNet -> TCN -> 2xLSTM(30) -> Fuzzy Neural
    Block (base paper model; falls back to trained-ml when no torch
    checkpoint is present)
  - ``logistic_regression``: 16-feature multinomial logreg (the only
    currently trained checkpoint -> correct appliance predictions)
  - ``eegnet``: EEGNet spatial-temporal conv net (same fallback)
  - ``eeg_conformer``: conv front-end + Transformer self-attention (same fallback)
  - ``cnn_bilstm``: conv extractor + bidirectional LSTM (same fallback)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.cnn_bilstm import CNNBiLSTMClassifier
from models.eeg_conformer import EEGConformerClassifier
from models.eeg_tcfnet import EEGTCFNetClassifier
from models.eegnet import TARGET_CLASSES, EEGNetClassifier
from models.logistic_regression import LogisticRegressionClassifier
from services.appliances import APPLIANCES, BY_NAME
from services.command_decoder import decode_command
from services.data_loader import (
    TARGET_FS,
    load_mat_file,
    load_upload,
    preprocess,
    detect_p300,
)

app = FastAPI(title="BCI Home Appliance Prediction (multi-model)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------- model catalog ---

MODEL_CATALOG: List[Dict[str, str]] = [
    {"id": "eeg_tcfnet", "name": "EEG-TCFNet (Base Paper Model)"},
    {"id": "eegnet", "name": "EEGNet (Spatial-Temporal Depthwise)"},
    {
        "id": "eeg_conformer",
        "name": "EEG-Conformer (CNN + Self-Attention Transformer)",
    },
    {"id": "cnn_bilstm", "name": "CNN + BiLSTM (Bidirectional Recurrent)"},
    {
        "id": "logistic_regression",
        "name": "Logistic Regression (16-Feature Baseline)",
    },
]
VALID_MODEL_IDS = [m["id"] for m in MODEL_CATALOG]
DEFAULT_MODEL = "eeg_tcfnet"

_classifiers: Dict[str, Any] = {}


def get_classifier(model_name: str):
    """Lazy singleton per model id (torch models stay in eval mode)."""
    if model_name not in _classifiers:
        if model_name == "eeg_tcfnet":
            _classifiers[model_name] = EEGTCFNetClassifier()
        elif model_name == "eegnet":
            _classifiers[model_name] = EEGNetClassifier()
        elif model_name == "eeg_conformer":
            _classifiers[model_name] = EEGConformerClassifier()
        elif model_name == "cnn_bilstm":
            _classifiers[model_name] = CNNBiLSTMClassifier()
        elif model_name == "logistic_regression":
            _classifiers[model_name] = LogisticRegressionClassifier()
        else:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Valid options: {VALID_MODEL_IDS}"
            )
    return _classifiers[model_name]


def _normalize_model_name(model_name: str) -> str:
    name = (model_name or DEFAULT_MODEL).strip().lower()
    if name not in VALID_MODEL_IDS:
        raise HTTPException(
            400,
            f"Unknown model_name '{model_name}'. "
            f"Valid options: {VALID_MODEL_IDS}",
        )
    return name


START_TIME = time.time()


def _preview(eeg: np.ndarray, n_channels: int = 4, n_points: int = 300) -> Dict[str, Any]:
    n_points = min(n_points, eeg.shape[1])
    idx = np.linspace(0, eeg.shape[1] - 1, n_points).astype(int)
    chs = min(n_channels, eeg.shape[0])
    return {
        "channels": [f"Ch{i + 1}" for i in range(chs)],
        "time": [round(float(t) / TARGET_FS, 4) for t in idx],
        "series": [
            {"channel": f"Ch{i + 1}", "values": [round(float(v), 4) for v in eeg[i, idx]]}
            for i in range(chs)
        ],
        "flat": [round(float(v), 4) for v in eeg[0, idx]],  # simple line-chart fallback
    }


def _predict_one(member: str, eeg_raw: np.ndarray, trig, key: str,
                 orig_fs: float, appliance_ctx: str = "",
                 model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    model_name = _normalize_model_name(model_name)
    prep = preprocess(eeg_raw, trig, orig_fs=orig_fs, source_key=key, member_name=member)
    classifier = get_classifier(model_name)
    out = classifier.predict(prep.eeg, eeg_raw=eeg_raw, trigger=trig, orig_fs=orig_fs)
    p300, p300_score = detect_p300(prep.eeg, prep.trigger)
    ctx_name = appliance_ctx if appliance_ctx in BY_NAME else out["predicted_appliance"]
    cmd = decode_command(
        prep.eeg, prep.trigger, orig_fs,
        BY_NAME.get(ctx_name, {}).get("commands", []),
    )
    if cmd is not None:
        cmd["file"] = member
        cmd["context_appliance"] = ctx_name
    out.update(
        {
            "file": member,
            "model_used": model_name,
            "p300_detected": bool(p300),
            "p300_score": round(float(p300_score), 4),
            "appliance_info": BY_NAME.get(out["predicted_appliance"]),
            "predicted_command": cmd,  # None when no trigger channel present
            "signal_preview": _preview(prep.eeg)["flat"],
            "signal_plot": _preview(prep.eeg),
            "meta": {
                **prep.meta,
                "orig_shape": list(prep.orig_shape),
                "orig_fs": prep.orig_fs,
                "source_key": prep.source_key,
            },
        }
    )
    return out


@app.get("/api/health")
def health() -> Dict[str, Any]:
    status = {}
    for mid in VALID_MODEL_IDS:
        try:
            clf = get_classifier(mid)
            status[mid] = {
                "loaded": True,
                "checkpoint_loaded": bool(getattr(clf, "checkpoint_loaded", False)),
                "has_torch": bool(getattr(clf, "model", None) is not None)
                or bool(getattr(clf, "__dict__", {}).get("model", None) is not None)
                or mid == "logistic_regression",
            }
        except Exception:
            status[mid] = {"loaded": False}
    return {
        "status": "on",  # "on" while the server is running; unreachable => off
        "model": f"multi-model (default: {DEFAULT_MODEL})",
        "active_default": DEFAULT_MODEL,
        "available_models": VALID_MODEL_IDS,
        "model_status": status,
        "classes": TARGET_CLASSES,
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


@app.get("/api/classes")
def classes() -> Dict[str, Any]:
    return {"classes": TARGET_CLASSES}


@app.get("/api/appliances")
def appliances() -> Dict[str, Any]:
    """Catalog of what the dataset contains for each appliance application."""
    return {"appliances": APPLIANCES}


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    """Model catalog for the frontend model-selection dropdown.

    Returns both the spec shape (``models``/``default``) and the legacy
    shape (``available_models``/``active_default``) so old and new
    frontends keep working.
    """
    return {
        "models": MODEL_CATALOG,
        "default": DEFAULT_MODEL,
        "available_models": MODEL_CATALOG,
        "active_default": DEFAULT_MODEL,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    orig_fs: float = 500.0,
    aggregate: str = "first",
    appliance: str = "",
    model_name: str = Query(DEFAULT_MODEL, description=(
        "Classifier to use: eeg_tcfnet | eegnet | eeg_conformer | "
        "cnn_bilstm | logistic_regression"
    )),
) -> JSONResponse:
    model_name = _normalize_model_name(model_name)
    if not file.filename or not file.filename.lower().endswith((".mat", ".zip")):
        raise HTTPException(400, "Upload must be a .mat file or .zip of .mat files.")
    try:
        blob = await file.read()
        items = load_upload(file.filename, blob)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse upload: {e}")

    try:
        if len(items) == 1 or aggregate == "first":
            name, eeg_raw, trig, key = items[0]
            result = _predict_one(name, eeg_raw, trig, key, orig_fs, appliance, model_name)
            result["files_processed"] = len(items)
            return JSONResponse(result)
        # aggregate == 'mean': average softmax over members
        probs = []
        previews = []
        first = None
        for name, eeg_raw, trig, key in items:
            r = _predict_one(name, eeg_raw, trig, key, orig_fs, appliance, model_name)
            if first is None:
                first = r
            probs.append([r["confidence_scores"][c] for c in TARGET_CLASSES])
            previews.append(name)
        mean_p = np.mean(np.array(probs), axis=0)
        mean_p = mean_p / (mean_p.sum() + 1e-12)
        cid = int(np.argmax(mean_p))
        return JSONResponse(
            {
                "predicted_appliance": TARGET_CLASSES[cid],
                "class_id": cid + 1,
                "confidence_scores": {c: round(float(p), 4) for c, p in zip(TARGET_CLASSES, mean_p)},
                "p300_detected": bool(first["p300_detected"]),
                "p300_score": first["p300_score"],
                "appliance_info": BY_NAME.get(TARGET_CLASSES[cid]),
                "predicted_command": first.get("predicted_command"),
                "signal_preview": first["signal_preview"],
                "signal_plot": first["signal_plot"],
                "meta": first["meta"],
                "files_processed": len(items),
                "files": previews,
                "model_used": model_name,
                "model_source": f"{first.get('model_source', '')} (softmax averaged over files)",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


@app.post("/api/predict_path")
def predict_path(
    path: str = Query(..., description="Server-local .mat path (dev only)"),
    model_name: str = Query(DEFAULT_MODEL, description=(
        "Classifier to use: eeg_tcfnet | eegnet | eeg_conformer | "
        "cnn_bilstm | logistic_regression"
    )),
):
    """Dev helper: run inference on a dataset file already on disk."""
    model_name = _normalize_model_name(model_name)
    try:
        eeg_raw, trig, key = load_mat_file(path)
    except Exception as e:
        raise HTTPException(400, f"Cannot load {path}: {e}")
    return JSONResponse(_predict_one(path, eeg_raw, trig, key, 500.0,
                                     model_name=model_name))


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "BCI Home Appliance Prediction (multi-model)",
        "docs": "/docs",
        "endpoints": ["POST /api/predict", "GET /api/health", "GET /api/classes",
                      "GET /api/appliances", "GET /api/models"],
    }

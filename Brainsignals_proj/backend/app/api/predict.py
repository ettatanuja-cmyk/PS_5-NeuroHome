"""Prediction and dataset endpoints (base paper model zoo).

  POST /api/predict          multipart ``file`` (.mat/.csv/.npy/.zip);
                             query ``aggregate=first|mean``, ``orig_fs``,
                             ``appliance=`` (optional label context),
                             ``model_name=`` (see GET /api/models)
  POST /api/predict_path     dev helper for server-local files
  GET  /api/health           status (dashboard StatusPill)
  GET  /api/classes          class list
  GET  /api/appliances       per-appliance catalog
  GET  /api/models           model-zoo catalog (dashboard ModelSelector)
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from ..models.common import (
    DEFAULT_MODEL,
    MODEL_CATALOG,
    TARGET_CLASSES,
    VALID_MODEL_IDS,
)
from ..models.eeg_tcfnet import EEGTCFNetClassifier
from ..models.eeg_tcnet import EEGTCNetClassifier, EEGTCNetLSTMClassifier
from ..models.lenet import LeNetClassifier
from ..models.traditional_ml import (
    LogisticRegressionClassifier,
    MLPClassifierZoo,
    SVMClassifier,
)
from ..preprocessing.signal_filters import TARGET_FS, detect_p300, preprocess
from ..utils.appliances import APPLIANCES, BY_NAME
from ..utils.command_decoder import decode_command
from ..utils.data_loader import load_mat_file, load_upload

router = APIRouter(prefix="/api", tags=["predict"])

START_TIME = time.time()

_BUILDERS = {
    "eeg_tcfnet": EEGTCFNetClassifier,
    "eeg_tcnet_lstm": EEGTCNetLSTMClassifier,
    "eeg_tcnet": EEGTCNetClassifier,
    "lenet": LeNetClassifier,
    "logistic_regression": LogisticRegressionClassifier,
    "svm": SVMClassifier,
    "mlp": MLPClassifierZoo,
}

_classifiers: Dict[str, Any] = {}


def get_classifier(model_id: str):
    """Lazy singleton per model id."""
    if model_id not in _classifiers:
        try:
            builder = _BUILDERS[model_id]
        except KeyError:
            raise HTTPException(
                400,
                f"Unknown model_name '{model_id}'. "
                f"Valid options: {VALID_MODEL_IDS}")
        _classifiers[model_id] = builder()
    return _classifiers[model_id]


def _check_model_name(model_name: str) -> str:
    name = (model_name or DEFAULT_MODEL).strip().lower()
    if name not in VALID_MODEL_IDS:
        raise HTTPException(
            400,
            f"Unknown model_name '{model_name}'. "
            f"Valid options: {VALID_MODEL_IDS}")
    return name


def _preview(eeg: np.ndarray, n_channels: int = 4,
             n_points: int = 300) -> Dict[str, Any]:
    n_points = min(n_points, eeg.shape[1])
    idx = np.linspace(0, eeg.shape[1] - 1, n_points).astype(int)
    chs = min(n_channels, eeg.shape[0])
    return {
        "channels": [f"Ch{i + 1}" for i in range(chs)],
        "time": [round(float(t) / TARGET_FS, 4) for t in idx],
        "series": [
            {"channel": f"Ch{i + 1}",
             "values": [round(float(v), 4) for v in eeg[i, idx]]}
            for i in range(chs)
        ],
        "flat": [round(float(v), 4) for v in eeg[0, idx]],
    }


def _predict_one(member: str, eeg_raw: np.ndarray, trig,
                 key: str, orig_fs: float, appliance_ctx: str = "",
                 model_id: str = DEFAULT_MODEL) -> Dict[str, Any]:
    prep = preprocess(eeg_raw, trig, orig_fs=orig_fs,
                      source_key=key, member_name=member)
    out = get_classifier(model_id).predict(prep.eeg)
    p300, p300_score = detect_p300(prep.eeg, prep.trigger)
    ctx_name = (appliance_ctx if appliance_ctx in BY_NAME
                else out["predicted_appliance"])
    cmd = decode_command(prep.eeg, prep.trigger, orig_fs,
                         BY_NAME.get(ctx_name, {}).get("commands", []))
    if cmd is not None:
        cmd["file"] = member
        cmd["context_appliance"] = ctx_name
    preview = _preview(prep.eeg)
    out.update({
        "file": member,
        "model_used": model_id,
        "p300_detected": bool(p300),
        "p300_score": round(float(p300_score), 4),
        "appliance_info": BY_NAME.get(out["predicted_appliance"]),
        "predicted_command": cmd,  # None when no trigger channel present
        "signal_preview": preview["flat"],
        "signal_plot": preview,
        "meta": {
            **prep.meta,
            "orig_shape": list(prep.orig_shape),
            "orig_fs": prep.orig_fs,
            "source_key": prep.source_key,
        },
    })
    return out


@router.get("/health")
def health() -> Dict[str, Any]:
    status = {}
    for mid in VALID_MODEL_IDS:
        try:
            clf = get_classifier(mid)
            status[mid] = {
                "loaded": True,
                "checkpoint_loaded": bool(getattr(clf, "checkpoint_loaded", False)),
                "has_torch": bool(getattr(clf, "model", None) is not None),
            }
        except Exception:
            status[mid] = {"loaded": False}
    default_clf = get_classifier(DEFAULT_MODEL)
    return {
        "status": "on",  # "on" while running; unreachable => off
        "model": f"model-zoo (default: {DEFAULT_MODEL})",
        "active_default": DEFAULT_MODEL,
        "available_models": MODEL_CATALOG,
        "model_status": status,
        "classes": TARGET_CLASSES,
        "checkpoint_loaded": bool(getattr(default_clf, "checkpoint_loaded", False)),
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


@router.get("/classes")
def classes() -> Dict[str, Any]:
    return {"classes": TARGET_CLASSES}


@router.get("/appliances")
def appliances() -> Dict[str, Any]:
    return {"appliances": APPLIANCES}


@router.get("/models")
def list_models() -> Dict[str, Any]:
    # Both shapes: {models, default} + legacy {available_models, active_default}
    return {
        "models": MODEL_CATALOG,
        "default": DEFAULT_MODEL,
        "available_models": MODEL_CATALOG,
        "active_default": DEFAULT_MODEL,
    }


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    orig_fs: float = 500.0,
    aggregate: str = "first",
    appliance: str = "",
    model_name: str = Query(DEFAULT_MODEL),
) -> JSONResponse:
    model_id = _check_model_name(model_name)
    if not file.filename or not file.filename.lower().endswith(
            (".mat", ".zip", ".csv", ".npy", ".npz")):
        raise HTTPException(
            400, "Upload must be a .mat/.csv/.npy file or .zip of those.")
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
            result = _predict_one(name, eeg_raw, trig, key, orig_fs,
                                  appliance, model_id)
            result["files_processed"] = len(items)
            return JSONResponse(result)
        # aggregate == 'mean': average softmax over members
        probs = []
        previews = []
        first = None
        for name, eeg_raw, trig, key in items:
            r = _predict_one(name, eeg_raw, trig, key, orig_fs,
                             appliance, model_id)
            if first is None:
                first = r
            probs.append([r["confidence_scores"][c] for c in TARGET_CLASSES])
            previews.append(name)
        assert first is not None
        mean_p = np.mean(np.array(probs), axis=0)
        mean_p = mean_p / (mean_p.sum() + 1e-12)
        cid = int(np.argmax(mean_p))
        return JSONResponse({
            "predicted_appliance": TARGET_CLASSES[cid],
            "class_id": cid + 1,
            "confidence_scores": {c: round(float(p), 4)
                                  for c, p in zip(TARGET_CLASSES, mean_p)},
            "p300_detected": bool(first["p300_detected"]),
            "p300_score": first["p300_score"],
            "appliance_info": BY_NAME.get(TARGET_CLASSES[cid]),
            "predicted_command": first.get("predicted_command"),
            "signal_preview": first["signal_preview"],
            "signal_plot": first["signal_plot"],
            "meta": first["meta"],
            "files_processed": len(items),
            "files": previews,
            "model_used": model_id,
            "model_source": (f"{first.get('model_source', '')} "
                             "(softmax averaged over files)"),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


@router.post("/predict_path")
def predict_path(
    path: str = Query(..., description="Server-local file (dev only)"),
    model_name: str = Query(DEFAULT_MODEL),
) -> JSONResponse:
    """Dev helper: run inference on a dataset file already on disk."""
    model_id = _check_model_name(model_name)
    try:
        eeg_raw, trig, key = load_mat_file(path)
    except Exception as e:
        raise HTTPException(400, f"Cannot load {path}: {e}")
    return JSONResponse(_predict_one(path, eeg_raw, trig, key, 500.0,
                                     model_id=model_id))

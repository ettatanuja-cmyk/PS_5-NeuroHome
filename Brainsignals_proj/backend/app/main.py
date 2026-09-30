"""EEG Smart Home Control backend (base paper model zoo).

Run from inside backend/:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.predict import DEFAULT_MODEL, router
from .models.common import MODEL_CATALOG, TARGET_CLASSES

app = FastAPI(title="EEG Smart Home Control (base paper model zoo)",
              version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "EEG Smart Home Control (base paper model zoo)",
        "default_model": DEFAULT_MODEL,
        "docs": "/docs",
        "endpoints": ["POST /api/predict", "GET /api/health",
                      "GET /api/classes", "GET /api/appliances",
                      "GET /api/models"],
        "models": MODEL_CATALOG,
        "classes": TARGET_CLASSES,
    }

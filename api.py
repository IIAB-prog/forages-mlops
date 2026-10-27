"""API FastAPI pour l'estimation du risque de panne des forages."""
from __future__ import annotations

import json
import logging
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import FEATURES, MODEL_DIR

LOGGER = logging.getLogger(__name__)
app = FastAPI(
    title="Forage Risk API",
    version="1.0.1",
    description="Estimation du risque de panne d'un forage dans les 90 jours.",
)

MODEL_PATH = MODEL_DIR / "forage_risk_model.joblib"
MODEL = None
MODEL_LOAD_ERROR: str | None = None


def load_model() -> None:
    """Charge le modèle sans empêcher l'API de démarrer en cas d'erreur."""
    global MODEL, MODEL_LOAD_ERROR
    try:
        if not MODEL_PATH.exists():
            MODEL = None
            MODEL_LOAD_ERROR = f"Fichier modèle absent : {MODEL_PATH}"
            LOGGER.error(MODEL_LOAD_ERROR)
            return
        MODEL = joblib.load(MODEL_PATH)
        MODEL_LOAD_ERROR = None
        LOGGER.info("Modèle chargé avec succès depuis %s", MODEL_PATH)
    except Exception as exc:
        MODEL = None
        MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Impossible de charger le modèle %s", MODEL_PATH)


@app.on_event("startup")
def startup_event() -> None:
    load_model()


class ForageInput(BaseModel):
    age_ans: int = Field(ge=0, le=100)
    profondeur_m: float = Field(gt=0, le=1000)
    debit_m3h: float = Field(ge=0, le=500)
    usagers_jour: int = Field(ge=0, le=100000)
    jours_depuis_maintenance: int = Field(ge=0, le=5000)
    pannes_12m: int = Field(ge=0, le=100)
    type_pompe: Literal["manuelle", "thermique", "solaire"]
    region: Literal["Centre", "Est", "Nord", "Sud"]


class Prediction(BaseModel):
    risque_panne_90j: float
    classe_risque: Literal["faible", "modere", "eleve"]
    seuil: float
    modele: str


@app.get("/health")
def health():
    """Endpoint diagnostic : l'API répond même si le modèle n'est pas chargeable."""
    return {
        "status": "ok" if MODEL is not None else "model_not_loaded",
        "model_loaded": MODEL is not None,
        "model_path": str(MODEL_PATH),
        "model_load_error": MODEL_LOAD_ERROR,
    }


@app.get("/metadata")
def metadata():
    path = MODEL_DIR / "metadata.json"
    if MODEL is None or not path.exists():
        raise HTTPException(
            status_code=503,
            detail={"message": "Modèle indisponible", "error": MODEL_LOAD_ERROR},
        )
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/predict", response_model=Prediction)
def predict(payload: ForageInput):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail={"message": "Modèle indisponible", "error": MODEL_LOAD_ERROR},
        )
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=503, detail="Métadonnées du modèle absentes")
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    risk = float(MODEL.predict_proba(row)[0, 1])
    class_risk = "eleve" if risk >= 0.70 else ("modere" if risk >= 0.40 else "faible")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    return Prediction(
        risque_panne_90j=round(risk, 4),
        classe_risque=class_risk,
        seuil=float(meta.get("threshold", 0.5)),
        modele=str(meta.get("model_name", "inconnu")),
    )

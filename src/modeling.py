"""Entraînement reproductible et comparaison des modèles candidats."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES, DATA_DIR, FEATURES, MINIMUM_AVERAGE_PRECISION,
    MODEL_DIR, NUMERIC_FEATURES, RANDOM_STATE, REPORT_DIR, TARGET, TEST_SIZE,
)

LOGGER = logging.getLogger(__name__)


def make_pipeline(model):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", prep), ("model", model)])


def calculate_metrics(y_true, pred, proba):
    return {
        "average_precision": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def _log_mlflow(name: str, metrics: dict, output_path: Path) -> None:
    """Trace paramètres/métriques/artifact sans sérialisation skops fragile."""
    if os.getenv("MLFLOW_TRACKING_URI"):
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    else:
        tracking_uri = "sqlite:///mlflow.db"
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("forages-mlops")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "random_state": RANDOM_STATE,
                               "test_size": TEST_SIZE, "threshold": 0.5})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            if output_path.exists():
                mlflow.log_artifact(str(output_path), artifact_path="candidate")
    except Exception as exc:  # MLflow indisponible ne doit pas casser l'entraînement
        LOGGER.warning("Traçabilité MLflow indisponible (%s).", exc)


def train(
    data_path: Path | None = None,
    model_output_path: Path | None = None,
    metadata_output_path: Path | None = None,
    write_reports: bool = True,
):
    """Entraîne plusieurs candidats et sauvegarde le meilleur vers le chemin demandé.

    Par défaut, conserve le comportement historique en sauvegardant dans models/.
    Le pipeline d'automatisation passe un chemin de candidat séparé pour ne jamais
    écraser le modèle en production avant les contrôles.
    """
    data_path = data_path or DATA_DIR / "forages_historique.csv"
    model_output_path = model_output_path or MODEL_DIR / "forage_risk_model.joblib"
    metadata_output_path = metadata_output_path or model_output_path.with_name("metadata.json")
    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible absente : {TARGET}")
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Variables absentes : {sorted(missing)}")
    df = df.dropna(subset=[TARGET]).copy()
    if df[TARGET].nunique() != 2:
        raise ValueError("La cible doit contenir les deux classes 0 et 1.")
    X, y = df[FEATURES], df[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    rows = []
    best = None
    best_score = -1.0
    for name, estimator in candidates.items():
        started = time.perf_counter()
        pipeline = make_pipeline(estimator)
        pipeline.fit(Xtr, ytr)
        probabilities = pipeline.predict_proba(Xte)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metric_values = calculate_metrics(yte, predictions, probabilities)
        metric_values.update({"run_name": name,
                              "fit_seconds": round(time.perf_counter() - started, 3),
                              "threshold": 0.5})
        rows.append(metric_values)
        if metric_values["average_precision"] > best_score:
            best_score = metric_values["average_precision"]
            best = (name, pipeline, metric_values, probabilities, predictions)
    if best is None:
        raise RuntimeError("Aucun modèle candidat n'a été entraîné.")
    name, pipeline, best_metrics, probabilities, predictions = best
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    metadata = {
        "model_name": name, "target": TARGET, "features": FEATURES,
        "threshold": 0.5, "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "metrics": best_metrics,
        "minimum_average_precision": MINIMUM_AVERAGE_PRECISION,
        "performance_gate_passed": best_metrics["average_precision"] >= MINIMUM_AVERAGE_PRECISION,
        "model_artifact": model_output_path.name,
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _log_mlflow(name, best_metrics, model_output_path)
    comparison = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    if write_reports:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
        pd.DataFrame({"y_true": yte.to_numpy(), "risk_probability": probabilities,
                      "prediction": predictions}).to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)
        (REPORT_DIR / "classification_report.txt").write_text(
            classification_report(yte, predictions, zero_division=0), encoding="utf-8"
        )
    return metadata, comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result, comparison = train()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(comparison.to_string(index=False))

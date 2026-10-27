"""Surveillance de dérive et des performances quand les étiquettes existent."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (
    CATEGORICAL_FEATURES, DATA_DIR, FEATURES, MINIMUM_AVERAGE_PRECISION,
    MODEL_DIR, NUMERIC_FEATURES, REPORT_DIR, TARGET,
)


def calculate_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        a, b = reference[col].dropna(), current[col].dropna()
        if len(a) == 0 or len(b) == 0:
            stat, p_value = 0.0, 1.0
        else:
            stat, p_value = ks_2samp(a, b)
        rows.append({"feature": col, "type": "numeric", "metric": "KS statistic",
                     "value": float(stat), "p_value": float(p_value),
                     "drift_flag": bool(p_value < 0.05)})
    for col in CATEGORICAL_FEATURES:
        pa = reference[col].fillna("<NA>").value_counts(normalize=True)
        pb = current[col].fillna("<NA>").value_counts(normalize=True)
        categories = pa.index.union(pb.index)
        distance = 0.5 * sum(abs(float(pa.get(c, 0)) - float(pb.get(c, 0))) for c in categories)
        rows.append({"feature": col, "type": "categorical",
                     "metric": "total_variation_distance", "value": float(distance),
                     "p_value": None, "drift_flag": bool(distance >= 0.10)})
    return pd.DataFrame(rows)


def monitor(reference_path: Path | None = None, current_path: Path | None = None,
            model_path: Path | None = None, write_reports: bool = True):
    reference_path = reference_path or DATA_DIR / "forages_historique.csv"
    current_path = current_path or DATA_DIR / "forages_nouveau_mois.csv"
    model_path = model_path or MODEL_DIR / "forage_risk_model.joblib"
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    for label, frame in (("référence", reference), ("courant", current)):
        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Variables absentes du jeu {label} : {sorted(missing)}")
    report = calculate_drift(reference, current)
    drift_count = int(report["drift_flag"].sum())
    severe_drift = drift_count >= 2
    performance = {"labeled_recent_rows": 0, "recent_average_precision": None,
                   "recent_roc_auc": None, "available": False}
    labeled = current.dropna(subset=[TARGET]) if TARGET in current.columns else current.iloc[0:0]
    if len(labeled) > 0 and labeled[TARGET].nunique() == 2 and model_path.exists():
        model = joblib.load(model_path)
        proba = model.predict_proba(labeled[FEATURES])[:, 1]
        performance = {
            "labeled_recent_rows": int(len(labeled)),
            "recent_average_precision": float(average_precision_score(labeled[TARGET].astype(int), proba)),
            "recent_roc_auc": float(roc_auc_score(labeled[TARGET].astype(int), proba)),
            "available": True,
        }
    low_performance = (performance["available"] and
                       performance["recent_average_precision"] < MINIMUM_AVERAGE_PRECISION)
    reasons = []
    if severe_drift:
        reasons.append("au moins deux variables en dérive")
    if low_performance:
        reasons.append("AP récente sous le seuil minimal")
    summary = {
        "reference_rows": int(len(reference)), "current_rows": int(len(current)),
        "drifted_features": drift_count, "total_features_checked": int(len(report)),
        "retraining_triggered": bool(severe_drift or low_performance),
        "trigger_reasons": reasons,
        "decision": "RETRAINING_REQUIRED" if reasons else "MONITOR_ONLY",
        "performance_on_labeled_recent_data": performance,
        "caveat": "La dérive ne prouve pas à elle seule une baisse de performance. Les labels doivent être fiables.",
    }
    if write_reports:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report.to_csv(REPORT_DIR / "drift_report.csv", index=False)
        (REPORT_DIR / "monitoring_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report, summary


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    result, summary = monitor()
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

"""Pipeline planifié : surveillance, réentraînement conditionnel, tests et promotion.

Exécution depuis la racine du dépôt : python -m src.pipeline --auto-promote
Sans --auto-promote, le candidat est évalué mais le modèle en production n'est pas remplacé.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config import (DATA_DIR, FEATURES, MINIMUM_AVERAGE_PRECISION, MODEL_DIR, REPORT_DIR, RANDOM_STATE, TARGET, TEST_SIZE)
from src.modeling import train
from src.monitoring import monitor

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODEL = MODEL_DIR / "forage_risk_model.joblib"
PRODUCTION_METADATA = MODEL_DIR / "metadata.json"
CANDIDATE_DIR = MODEL_DIR / "candidates"
LEDGER = REPORT_DIR / "automation_ledger.jsonl"
LOGGER = logging.getLogger("forages.pipeline")


def _write_event(event: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with LEDGER.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def _run_tests() -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Les tests automatisés ont échoué (code {result.returncode}).")


def _candidate_passes(candidate: dict, baseline: dict | None) -> tuple[bool, list[str]]:
    reasons = []
    candidate_metrics = candidate.get("metrics", {})
    candidate_ap = candidate_metrics.get("average_precision")
    candidate_auc = candidate_metrics.get("roc_auc")
    if candidate_ap is None or candidate_ap < MINIMUM_AVERAGE_PRECISION:
        reasons.append(f"AP candidat inférieure au seuil {MINIMUM_AVERAGE_PRECISION:.2f}")
    if not candidate.get("performance_gate_passed", False):
        reasons.append("garde de performance minimale non satisfaite")
    if baseline:
        baseline_metrics = baseline.get("metrics", {})
        baseline_ap = baseline_metrics.get("average_precision")
        baseline_auc = baseline_metrics.get("roc_auc")
        # Tolérances modestes pour éviter de promouvoir une régression substantielle.
        if baseline_ap is not None and candidate_ap is not None and candidate_ap < baseline_ap - 0.01:
            reasons.append("AP du candidat en baisse de plus de 0,01 par rapport au modèle actif")
        if baseline_auc is not None and candidate_auc is not None and candidate_auc < baseline_auc - 0.02:
            reasons.append("ROC AUC du candidat en baisse de plus de 0,02 par rapport au modèle actif")
    return not reasons, reasons


def _promote_and_deploy(candidate_path: Path, candidate_metadata_path: Path, metadata: dict) -> None:
    """Promouvoir le candidat, vérifier l'API puis restaurer les artefacts en cas d'échec."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = MODEL_DIR / "rollback"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_model = backup_dir / f"model_{stamp}.joblib"
    backup_metadata = backup_dir / f"metadata_{stamp}.json"

    had_previous_model = PRODUCTION_MODEL.exists()
    had_previous_metadata = PRODUCTION_METADATA.exists()
    if had_previous_model:
        shutil.copy2(PRODUCTION_MODEL, backup_model)
    if had_previous_metadata:
        shutil.copy2(PRODUCTION_METADATA, backup_metadata)

    new_metadata = dict(metadata)
    new_metadata["model_artifact"] = PRODUCTION_MODEL.name
    staged_model = MODEL_DIR / "forage_risk_model.staged.joblib"
    staged_metadata = MODEL_DIR / "metadata.staged.json"
    shutil.copy2(candidate_path, staged_model)
    staged_metadata.write_text(
        json.dumps(new_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    staged_model.replace(PRODUCTION_MODEL)
    staged_metadata.replace(PRODUCTION_METADATA)

    try:
        # Rebuild indispensable après modification des dépendances (scikit-learn).
        # --force-recreate garantit que le processus recharge l'artefact monté.
        subprocess.run(
            ["docker", "compose", "up", "-d", "--build", "--force-recreate", "--no-deps", "api"],
            cwd=ROOT,
            check=True,
        )
        _wait_for_health()
        LOGGER.info("Nouvelle version de l'API déployée et contrôle de santé validé.")
    except Exception:
        LOGGER.exception("Échec du déploiement : restauration du modèle précédent.")
        if had_previous_model and backup_model.exists():
            shutil.copy2(backup_model, PRODUCTION_MODEL)
        elif not had_previous_model and PRODUCTION_MODEL.exists():
            PRODUCTION_MODEL.unlink()
        if had_previous_metadata and backup_metadata.exists():
            shutil.copy2(backup_metadata, PRODUCTION_METADATA)
        elif not had_previous_metadata and PRODUCTION_METADATA.exists():
            PRODUCTION_METADATA.unlink()

        # Reconstruire/redémarrer sur les artefacts restaurés. Ne pas masquer l'erreur initiale.
        rollback = subprocess.run(
            ["docker", "compose", "up", "-d", "--force-recreate", "--no-deps", "api"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if rollback.returncode != 0:
            LOGGER.error("Échec du redémarrage après rollback : %s", rollback.stderr)
        else:
            try:
                _wait_for_health()
                LOGGER.info("Rollback confirmé : l'API précédente est saine.")
            except Exception:
                LOGGER.exception("Le rollback des fichiers a été fait, mais l'API n'est pas saine.")
        raise


def _wait_for_health(timeout_seconds: int = 90, interval_seconds: int = 3) -> None:
    """Attend l'API publiée sur localhost:8000 et exige le chargement du modèle."""
    import time
    import urllib.error
    import urllib.request

    url = "http://127.0.0.1:8000/health"
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | str = "aucune réponse reçue"

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and payload.get("model_loaded") is True:
                return
            last_error = (
                f"Healthcheck non valide : {payload}. "
                "Consulter `docker compose logs --tail=150 api`."
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(interval_seconds)

    # Ajouter les journaux Docker à l'erreur pour diagnostiquer sans étape manuelle.
    logs = subprocess.run(
        ["docker", "compose", "logs", "--tail=80", "api"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    log_tail = (logs.stdout or "") + (logs.stderr or "")
    raise RuntimeError(
        f"L'API n'est pas saine après {timeout_seconds}s : {last_error}. "
        f"Journaux Docker récents :\\n{log_tail[-12000:]}"
    )

def run_pipeline(auto_promote: bool = False) -> dict:
    event = {"status": "started", "auto_promote": auto_promote}
    try:
        _, monitoring = monitor()
        event["monitoring"] = monitoring
        if not monitoring["retraining_triggered"] and PRODUCTION_MODEL.exists():
            event.update({"status": "completed", "action": "monitor_only"})
            _write_event(event)
            return event

        current_data_path = DATA_DIR / "forages_nouveau_mois.csv"
        current_frame = pd.read_csv(current_data_path)
        labeled_rows = current_frame.dropna(subset=[TARGET]) if TARGET in current_frame.columns else current_frame.iloc[0:0]
        if PRODUCTION_MODEL.exists() and (len(labeled_rows) < 30 or labeled_rows[TARGET].nunique() != 2):
            event.update({
                "status": "completed",
                "action": "awaiting_reliable_labels",
                "labeled_recent_rows": int(len(labeled_rows)),
                "message": "Dérive détectée, mais pas assez de labels récents fiables pour réentraîner et valider un candidat. Le modèle actif est conservé.",
            })
            _write_event(event)
            return event

        CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_path = CANDIDATE_DIR / f"forage_risk_candidate_{stamp}.joblib"
        candidate_metadata_path = CANDIDATE_DIR / f"metadata_{stamp}.json"
        training_path = DATA_DIR / "processed" / f"training_dataset_{stamp}.csv"
        training_path.parent.mkdir(parents=True, exist_ok=True)
        historical = pd.read_csv(DATA_DIR / "forages_historique.csv")
        # N'utiliser les données du mois que si leurs résultats réels sont disponibles.
        if len(labeled_rows) >= 30 and labeled_rows[TARGET].nunique() == 2:
            training_frame = pd.concat([historical, labeled_rows], ignore_index=True)
        else:
            training_frame = historical
        training_frame.to_csv(training_path, index=False)
        event["training_rows"] = int(len(training_frame))
        event["recent_labeled_rows_used"] = int(len(labeled_rows))
        candidate, comparison = train(
            data_path=training_path,
            model_output_path=candidate_path,
            metadata_output_path=candidate_metadata_path,
            write_reports=False,
        )
        comparison.to_csv(REPORT_DIR / "candidate_comparison.csv", index=False)
        _run_tests()
        baseline = None
        # Comparer candidat et modèle actif sur exactement le même holdout temporellement
        # neutre (split reproductible) pour éviter de comparer des métriques de jeux différents.
        evaluation_frame = training_frame.dropna(subset=[TARGET]).copy()
        X_eval, y_eval = evaluation_frame[FEATURES], evaluation_frame[TARGET].astype(int)
        _, X_holdout, _, y_holdout = train_test_split(
            X_eval, y_eval, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y_eval,
        )
        candidate_model = joblib.load(candidate_path)
        candidate_proba = candidate_model.predict_proba(X_holdout)[:, 1]
        candidate_holdout_metrics = {
            "average_precision": float(average_precision_score(y_holdout, candidate_proba)),
            "roc_auc": float(roc_auc_score(y_holdout, candidate_proba)),
        }
        if PRODUCTION_MODEL.exists():
            baseline_model = joblib.load(PRODUCTION_MODEL)
            baseline_proba = baseline_model.predict_proba(X_holdout)[:, 1]
            baseline = {"metrics": {
                "average_precision": float(average_precision_score(y_holdout, baseline_proba)),
                "roc_auc": float(roc_auc_score(y_holdout, baseline_proba)),
            }}
        passed, reasons = _candidate_passes(
            {**candidate, "metrics": candidate_holdout_metrics,
             "performance_gate_passed": candidate_holdout_metrics["average_precision"] >= MINIMUM_AVERAGE_PRECISION},
            baseline,
        )
        event.update({
            "candidate_path": str(candidate_path.relative_to(ROOT)),
            "candidate_metrics": candidate.get("metrics"),
            "same_holdout_candidate_metrics": candidate_holdout_metrics,
            "same_holdout_active_model_metrics": baseline.get("metrics") if baseline else None,
            "quality_gate_passed": passed,
            "quality_gate_reasons": reasons,
            "tests_passed": True,
        })
        if not passed:
            event.update({"status": "completed", "action": "candidate_rejected"})
        elif not auto_promote:
            event.update({"status": "completed", "action": "candidate_approved_pending_promotion"})
        else:
            _promote_and_deploy(candidate_path, candidate_metadata_path, candidate)
            event.update({"status": "completed", "action": "promoted_and_deployed"})
    except Exception as exc:
        LOGGER.exception("Échec du pipeline MLOps.")
        event.update({"status": "failed", "error": str(exc)})
        _write_event(event)
        raise
    _write_event(event)
    return event


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline d'automatisation MLOps forages")
    parser.add_argument("--auto-promote", action="store_true",
                        help="Promouvoir le candidat s'il passe les tests et les garde-fous, puis redémarrer l'API Docker.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_pipeline(auto_promote=args.auto_promote)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

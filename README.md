# Forages MLOps — prédiction du risque de panne à 90 jours

Projet pédagogique complet : entraînement reproductible, comparaison de modèles, API FastAPI, tests, Docker, CI et surveillance de dérive. Les données fournies sont fictives. La sortie est une aide à la priorisation, pas une décision automatique de maintenance.

## 1. Données et cible
- `data/forages_historique.csv` : 3 000 lignes pour entraîner/évaluer.
- `data/forages_nouveau_mois.csv` : 600 lignes récentes pour le scoring et la surveillance.
- Cible d’apprentissage : `panne_90j` (0/1). Elle n’est jamais utilisée comme variable d’entrée.
- Variables : âge, profondeur, débit, usagers/jour, jours depuis maintenance, pannes sur 12 mois, type de pompe, région.

## 2. Installation (Python 3.11 recommandé)
```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Exécution
```bash
python -m src.modeling
python -m src.monitoring
python -m src.pipeline                 # surveillance + candidat, sans promotion automatique
python -m src.pipeline --auto-promote  # promouvoir si tests/garde-fous passent, puis redémarrer Docker Compose
python predict_batch.py
pytest -q
uvicorn api:app --reload
```
Ouvrir `http://127.0.0.1:8000/docs`. Exemple JSON de `/predict` :
```json
{"age_ans":15,"profondeur_m":90,"debit_m3h":4.2,"usagers_jour":450,"jours_depuis_maintenance":300,"pannes_12m":1,"type_pompe":"solaire","region":"Centre"}
```

## 4. MLflow
L’entraînement exécute trois essais. MLflow enregistre paramètres, métriques et artifact candidat dans une base SQLite `mlflow.db` (URI `MLFLOW_TRACKING_URI` configurable). Pour consulter : `mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000`. Si MLflow est indisponible, l’entraînement continue avec un avertissement ; la comparaison CSV reste disponible.

## 5. Docker
```bash
docker compose up --build
```
L’image expose l’API sur le port 8000. Le dossier `models/` de l’hôte est monté en lecture seule dans le conteneur afin que la promotion contrôlée du modèle soit visible après redémarrage, sans réentraîner au build.

## 6. Surveillance et règle opérationnelle
`src/monitoring.py` compare les variables numériques (test KS, seuil p < 0,05) et catégorielles (distance de variation totale, seuil 0,10). `src/pipeline.py` peut être exécuté périodiquement : au moins deux variables en dérive ou une AP récente sous 0,50 déclenchent l’évaluation d’un réentraînement. Le candidat est entraîné séparément, les tests sont exécutés et les seuils sont comparés au modèle actif. Sans `--auto-promote`, aucune promotion n’est effectuée. Avec `--auto-promote`, le modèle n’est promu que si les garde-fous passent ; l’API Docker est redémarrée et contrôlée, avec restauration du modèle précédent si le déploiement échoue. Les données mensuelles ne sont ajoutées à l’entraînement que si la cible est présente et contient les deux classes. Les données de démonstration sont synthétiques : les seuils doivent être validés avant usage réel.

## 7. Seuils et limites
Le seuil de classification est fixé à 0,50 pour la démonstration; les classes d’interface sont faible (<0,40), modéré (0,40–0,70), élevé (>=0,70). Ces seuils doivent être calibrés selon le coût d’une panne et la capacité de maintenance. Le jeu récent contient une cible simulée permettant un contrôle hors production, mais cette cible ne doit pas être disponible au moment du scoring réel.

## 8. Arborescence
- `src/modeling.py` : prétraitement, trois modèles, métriques, sauvegarde.
- `api.py` : endpoints `/health`, `/metadata`, `/predict`.
- `src/monitoring.py` : rapport de dérive et décision de surveillance.
- `src/pipeline.py` : orchestration conditionnelle, tests, garde-fous de promotion et rollback.
- `scripts/run_pipeline.bat` et `scripts/install_scheduled_task.ps1` : planification Windows quotidienne à 02:00.
- `predict_batch.py` : scores des 600 forages récents.
- `tests/` : tests de contrat et de dérive.
- `.github/workflows/ci.yml` : intégration continue.
- `reports/` : résultats générés à l’exécution.

## 9. Reproductibilité
Graine aléatoire 42, séparation stratifiée 80/20, pipeline scikit-learn, transformations ajustées uniquement sur le jeu d’entraînement. Toutes les métriques sont générées par les scripts du dépôt.

## 10. Planification automatique sous Windows
1. Créer l'environnement `.venv` et installer les dépendances (`pip install -r requirements.txt`).
2. Démarrer Docker Desktop et vérifier `docker compose up -d --build`.
3. Ouvrir PowerShell dans le dépôt et exécuter `Set-ExecutionPolicy -Scope Process Bypass`, puis `./scripts/install_scheduled_task.ps1`.
4. La tâche `Forages-MLOps-Pipeline` s'exécute chaque jour à 02:00 sous le compte Windows courant. Docker Desktop doit être accessible à ce compte.
5. Vérifier `reports/automation_ledger.jsonl`, `reports/drift_report.csv` et `reports/monitoring_summary.json` après chaque exécution.
6. Tester d'abord sans déploiement avec `python -m src.pipeline`. L'option `--auto-promote` active la promotion automatique, mais seulement après les tests, les seuils de performance et le contrôle de santé de l'API.

**Limite importante :** les données fournies sont synthétiques et possèdent déjà la cible `panne_90j`. En production, ne pas considérer les labels simulés comme des résultats réels. Pour une panne à horizon 90 jours, les performances ne peuvent être calculées qu'après collecte et validation du résultat réel. Sans assez de labels récents, le pipeline conserve le modèle actif et attend de nouvelles étiquettes. La dérive des variables seule n'est pas une preuve de baisse de performance.
\n\n## Dépannage de l'API Docker\nLe dépôt comprend `Dockerfile` et `docker-compose.yml`. L'image publie `8000:8000`. La dépendance `scikit-learn` est fixée à `1.6.1` pour rester compatible avec le modèle joblib fourni. Après modification des dépendances, reconstruire l'image avec `docker compose up -d --build --force-recreate api`. Vérifier `docker compose port api 8000` et `docker compose logs --tail=100 api`. L'endpoint `/health` indique `model_load_error` si le modèle ne peut pas être chargé.\n
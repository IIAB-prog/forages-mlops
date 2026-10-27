# Résumé d'exécution — projet Forages MLOps

- Données historiques : 3 000 lignes, 9 colonnes.
- Données du mois récent : 600 lignes, 9 colonnes.
- Cible : `panne_90j`; exclue des variables d'entrée.
- Entraînement : séparation stratifiée 80/20, graine 42.
- Trois essais comparés : régression logistique, Extra Trees, Random Forest.
- Modèle retenu par average precision : logistic_regression.
- Métriques hold-out : AP 0,606 ; ROC AUC 0,751 ; rappel 0,679 ; précision 0,558 ; F1 0,613 ; accuracy 0,688.
- Garde-fou AP >= 0,50 : passé.
- Surveillance : dérive statistique sur `usagers_jour` et `pannes_12m`; décision = évaluer un ré-entraînement et valider avec le métier.
- Jeu récent simulé : AP 0,739 et ROC AUC 0,780, à interpréter avec prudence.
- Tests exécutés : 4 réussis.

## Point de transparence sur MLflow
Le code est intégré pour consigner automatiquement les trois essais dans MLflow si la dépendance est installée. Dans l'environnement de génération, MLflow n'était pas installé : les trois essais sont donc effectivement documentés dans `model_comparison.csv` / `run_ledger.csv`, mais aucune expérience MLflow locale n'a été créée ici. Après `pip install -r requirements.txt`, relancer `python -m src.modeling` puis `mlflow ui --backend-store-uri ./mlruns` pour produire et consulter les runs MLflow.

Les données étant fictives, les scores ne doivent pas être utilisés pour une décision réelle sans validation sur des données terrain.

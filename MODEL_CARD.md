# Model Card — Forage Risk v1.0

## Usage prévu
Prioriser les visites de maintenance préventive en estimant la probabilité de panne dans les 90 jours. Le modèle ne doit pas décider seul d'une intervention.

## Données
- Entraînement/évaluation : 3000 lignes simulées, 36.3% de pannes positives.
- Jeu récent : 600 lignes simulées, utilisé pour surveillance et démonstration.
- Variables : âge, profondeur, débit, usagers/jour, jours depuis maintenance, pannes 12 mois, type de pompe, région.
- Cible : `panne_90j`. La cible n'est pas une entrée du modèle.

## Modèle retenu
logistic_regression dans un pipeline scikit-learn avec imputation, standardisation des numériques et one-hot encoding des catégorielles. Graine=42; split stratifié 20% test.

## Résultats hold-out
- Average precision : 0.606
- ROC AUC : 0.751
- Précision à seuil 0,5 : 0.558
- Rappel à seuil 0,5 : 0.679
- F1 : 0.613
- Exactitude : 0.688

## Garde-fou
Seuil de validation pédagogique AP >= 0.50 : PASSÉ. Le seuil doit être justifié en fonction des coûts de faux négatifs/faux positifs.

## Limites et risques
Données fictives et potentiellement simplifiées; aucune garantie de généralisation au terrain. Les distributions et taux de panne peuvent varier par région, type de pompe ou accès à la maintenance. Les probabilités ne sont pas nécessairement calibrées. Contrôle humain et validation terrain requis avant tout usage opérationnel.

## Surveillance
Comparaison statistique des variables de référence et du mois récent; dérive détectée sur usagers_jour, pannes_12m. La dérive déclenche une investigation, pas un ré-entraînement automatique.

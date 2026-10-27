from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TARGET = "panne_90j"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MINIMUM_AVERAGE_PRECISION = 0.50
NUMERIC_FEATURES = ["age_ans", "profondeur_m", "debit_m3h", "usagers_jour", "jours_depuis_maintenance", "pannes_12m"]
CATEGORICAL_FEATURES = ["type_pompe", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

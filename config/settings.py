from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REAL_DATA_DIR = ROOT_DIR / "new data" / "DATA"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
ASSET_DIR = ROOT_DIR / "assets"

RANDOM_STATE = 42
N_PATIENTS = 2600

HISTORICAL_END = "2024-06-30"
CURRENT_START = "2024-07-01"
CURRENT_END = "2026-03-31"

TARGET = "diabetes_present"
ID_COLUMNS = ["patient_id", "index_date"]

MODEL_SEARCH = {
    "Decision Tree": {
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": [8, 16, 32],
        "model__class_weight": [None, "balanced"],
    },
    "SVM": {
        "model__C": [0.5, 1.0, 2.0],
        "model__gamma": ["scale", 0.05],
        "model__class_weight": [None, "balanced"],
    },
    "Neural Network": {
        "model__hidden_layer_sizes": [(48,), (64, 24)],
        "model__alpha": [0.0005, 0.002],
        "model__learning_rate_init": [0.001],
    },
}

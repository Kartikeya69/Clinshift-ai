from __future__ import annotations

import joblib
import pandas as pd
from sklearn.base import clone

from config.settings import MODEL_DIR
from src.modeling import evaluate_model, split_xy


def continual_retrain(best_name: str, model, historical_train: pd.DataFrame, current_train: pd.DataFrame, current_test: pd.DataFrame, target: str, model_dir=MODEL_DIR) -> tuple[object, pd.DataFrame]:
    before = evaluate_model(model, current_test, "Before Continual Learning", target)
    expanded = pd.concat([historical_train, current_train], ignore_index=True)
    tuned = clone(model)
    x, y = split_xy(expanded, target)
    tuned.fit(x, y)
    after = evaluate_model(tuned, current_test, "After Continual Learning", target)
    for row in (before, after):
        row["model"] = best_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned, model_dir / "continual_best_model.joblib")
    return tuned, pd.DataFrame([before, after])

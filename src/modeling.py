from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config.settings import MODEL_DIR, MODEL_SEARCH, RANDOM_STATE, TARGET
from src.features import feature_columns
from utils.io import ensure_dirs


def split_xy(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    return df[feature_columns(df, target)], df[target].astype(int)


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def model_registry() -> dict[str, object]:
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Neural Network": MLPClassifier(max_iter=450, early_stopping=True, random_state=RANDOM_STATE),
    }


def train_models(train_df: pd.DataFrame, target: str = TARGET, model_dir=MODEL_DIR) -> dict[str, Pipeline]:
    ensure_dirs(model_dir)
    x_train, y_train = split_xy(train_df, target)
    trained: dict[str, Pipeline] = {}
    for name, estimator in model_registry().items():
        pipe = Pipeline([("preprocess", make_preprocessor(x_train)), ("model", estimator)])
        search = RandomizedSearchCV(
            pipe,
            MODEL_SEARCH[name],
            n_iter=min(6, np.prod([len(v) for v in MODEL_SEARCH[name].values()])),
            scoring="roc_auc",
            cv=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        trained[name] = search.best_estimator_
        joblib.dump(search.best_estimator_, model_dir / f"{name.lower().replace(' ', '_')}.joblib")
    return trained


def predict_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


def evaluate_model(model: Pipeline, df: pd.DataFrame, cohort: str, target: str = TARGET) -> dict:
    x, y = split_xy(df, target)
    proba = predict_scores(model, x)
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "cohort": cohort,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_all(models: dict[str, Pipeline], historical_test: pd.DataFrame, current_test: pd.DataFrame, historical_train: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        for cohort, df in [("Historical Train", historical_train), ("Historical Test", historical_test), ("Current Test", current_test)]:
            row = evaluate_model(model, df, cohort, target)
            row["model"] = name
            rows.append(row)
    metrics = pd.DataFrame(rows)
    hist = metrics[metrics["cohort"] == "Historical Test"][["model", "roc_auc", "f1"]].rename(columns={"roc_auc": "hist_auc", "f1": "hist_f1"})
    cur = metrics[metrics["cohort"] == "Current Test"][["model", "roc_auc", "f1"]].rename(columns={"roc_auc": "current_auc", "f1": "current_f1"})
    deltas = hist.merge(cur, on="model")
    deltas["auc_degradation"] = deltas["hist_auc"] - deltas["current_auc"]
    deltas["f1_degradation"] = deltas["hist_f1"] - deltas["current_f1"]
    return metrics.merge(deltas[["model", "auc_degradation", "f1_degradation"]], on="model", how="left")


def load_models(model_dir=MODEL_DIR) -> dict[str, Pipeline]:
    models = {}
    for path in model_dir.glob("*.joblib"):
        name = path.stem.replace("_", " ").title().replace("Svm", "SVM")
        models[name] = joblib.load(path)
    return models


def get_feature_names(model: Pipeline) -> list[str]:
    prep = model.named_steps["preprocess"]
    names = prep.get_feature_names_out()
    return [name.replace("num__", "").replace("cat__", "") for name in names]


def feature_importance(model: Pipeline, train_df: pd.DataFrame, target: str = TARGET, top_n: int = 18) -> pd.DataFrame:
    x_train, _ = split_xy(train_df, target)
    names = get_feature_names(model)
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coefs_"):
        values = np.mean(np.abs(estimator.coefs_[0]), axis=1)
    else:
        sample = x_train.sample(min(400, len(x_train)), random_state=RANDOM_STATE)
        transformed = model.named_steps["preprocess"].transform(sample)
        baseline = predict_scores(model, sample)
        values = []
        rng = np.random.default_rng(RANDOM_STATE)
        arr = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        for idx in range(arr.shape[1]):
            shuffled = arr.copy()
            rng.shuffle(shuffled[:, idx])
            est = estimator
            scores = est.predict_proba(shuffled)[:, 1]
            values.append(float(np.mean(np.abs(baseline - scores))))
        values = np.asarray(values)
    out = pd.DataFrame({"feature": names[: len(values)], "importance": np.abs(values)})
    return out.sort_values("importance", ascending=False).head(top_n)

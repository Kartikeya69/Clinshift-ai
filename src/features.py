from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.diseases import all_targets
from config.settings import CURRENT_START, HISTORICAL_END, ID_COLUMNS, PROCESSED_DATA_DIR, RAW_DATA_DIR, RANDOM_STATE, TARGET
from utils.io import ensure_dirs


def load_raw_tables() -> dict[str, pd.DataFrame]:
    parse = {
        "patients": ["registration_date"],
        "encounters": ["visit_date"],
        "observations": ["observation_date"],
        "conditions": ["diagnosis_date"],
    }
    return {
        name: pd.read_csv(RAW_DATA_DIR / f"{name}.csv", parse_dates=dates)
        for name, dates in parse.items()
    }


def _wide_observations(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.drop_duplicates(["patient_id", "encounter_id", "parameter", "observation_date"])
    pivot = observations.pivot_table(
        index="patient_id",
        columns="parameter",
        values="value",
        aggfunc=["mean", "median", "std", "min", "max", "count"],
    )
    pivot.columns = [f"{param}_{stat}" for stat, param in pivot.columns]
    return pivot.reset_index()


def build_feature_matrix(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    tables = tables or load_raw_tables()
    patients = tables["patients"].drop_duplicates("patient_id").copy()
    encounters = tables["encounters"].drop_duplicates("encounter_id").copy()
    observations = tables["observations"].copy()
    conditions = tables["conditions"].drop_duplicates(["patient_id", "condition_code", "diagnosis_date"]).copy()

    obs_features = _wide_observations(observations)
    encounter_agg = encounters.groupby("patient_id").agg(
        visit_count=("encounter_id", "nunique"),
        first_visit=("visit_date", "min"),
        last_visit=("visit_date", "max"),
        urgent_visit_rate=("visit_type", lambda s: float((s == "Urgent").mean())),
        telehealth_rate=("visit_type", lambda s: float((s == "Telehealth").mean())),
        specialty_rate=("visit_type", lambda s: float((s == "Endocrinology").mean())),
    ).reset_index()

    condition_text = conditions["condition_name"].fillna("").astype(str)
    condition_flags = pd.DataFrame({"patient_id": conditions["patient_id"].drop_duplicates()})
    flag_defs = {
        "cond_hypertension": "hypertension|high blood pressure",
        "cond_obesity": "obesity|body mass index",
        "cond_hyperlipidemia": "hyperlipidemia|cholesterol",
        "cond_diabetes": "diabetes|diabetic",
        "cond_prediabetes": "prediabetes",
    }
    for name, pattern in flag_defs.items():
        flagged = conditions.loc[condition_text.str.contains(pattern, case=False, na=False), ["patient_id"]].drop_duplicates()
        flagged[name] = 1
        condition_flags = condition_flags.merge(flagged, on="patient_id", how="left")

    df = patients.merge(obs_features, on="patient_id", how="left").merge(encounter_agg, on="patient_id", how="left")
    df = df.merge(condition_flags, on="patient_id", how="left")
    for col in ["cond_hypertension", "cond_obesity", "cond_hyperlipidemia", "cond_diabetes", "cond_prediabetes"]:
        if col not in df:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    df["index_date"] = df["last_visit"].fillna(df["registration_date"])
    df["time_since_last_visit_days"] = (pd.Timestamp("2026-04-01") - df["last_visit"]).dt.days
    df["care_span_days"] = (df["last_visit"] - df["first_visit"]).dt.days.clip(lower=1)
    df["visits_per_year"] = df["visit_count"] / (df["care_span_days"] / 365.25)
    for col in [
        "height_cm_mean",
        "weight_kg_mean",
        "systolic_bp_mean",
        "diastolic_bp_mean",
        "glucose_mean",
        "glucose_std",
        "hba1c_mean",
        "cholesterol_mean",
    ]:
        if col not in df:
            df[col] = np.nan
    calculated_bmi = df["weight_kg_mean"] / ((df["height_cm_mean"] / 100) ** 2)
    if "bmi_mean" in df:
        df["bmi_mean"] = df["bmi_mean"].combine_first(calculated_bmi)
    else:
        df["bmi_mean"] = calculated_bmi
    df["bp_mean"] = (df["systolic_bp_mean"] + 2 * df["diastolic_bp_mean"]) / 3
    df["glucose_variability"] = df["glucose_std"].fillna(0) / df["glucose_mean"].replace(0, np.nan)
    df["metabolic_syndrome_score"] = (
        (df["bmi_mean"] >= 30).astype(int)
        + (df["glucose_mean"] >= 126).astype(int)
        + (df["systolic_bp_mean"] >= 130).astype(int)
        + (df["cholesterol_mean"] >= 200).astype(int)
    )
    df["high_risk_flag"] = ((df["metabolic_syndrome_score"] >= 2) | (df["hba1c_mean"] >= 6.5)).astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 39, 64, 120],
        labels=["young_adult", "adult", "geriatric"],
        include_lowest=True,
    ).astype(str)
    df["period"] = np.where(df["index_date"] <= pd.Timestamp(HISTORICAL_END), "historical", "current")

    date_cols = ["registration_date", "first_visit", "last_visit"]
    return df.drop(columns=date_cols).replace([np.inf, -np.inf], np.nan)


def make_temporal_splits(features: pd.DataFrame, target: str = TARGET) -> dict[str, pd.DataFrame]:
    historical = features[features["period"] == "historical"].copy()
    current = features[features["index_date"] >= pd.Timestamp(CURRENT_START)].copy()

    def stratify_or_none(df: pd.DataFrame):
        counts = df[target].value_counts()
        return df[target] if len(counts) > 1 and counts.min() >= 2 else None

    hist_train, hist_test = train_test_split(
        historical,
        test_size=0.28,
        random_state=RANDOM_STATE,
        stratify=stratify_or_none(historical),
    )
    current_train, current_test = train_test_split(
        current,
        test_size=0.55,
        random_state=RANDOM_STATE,
        stratify=stratify_or_none(current),
    )
    return {
        "historical_train": hist_train,
        "historical_test": hist_test,
        "current_train": current_train,
        "current_test": current_test,
    }


def save_processed(features: pd.DataFrame, splits: dict[str, pd.DataFrame] | None = None, base_dir=PROCESSED_DATA_DIR) -> None:
    ensure_dirs(base_dir)
    features.to_csv(base_dir / "features.csv", index=False)
    if splits:
        for name, split in splits.items():
            split.to_csv(base_dir / f"{name}.csv", index=False)


def feature_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    targets = set(all_targets())
    if target:
        targets.add(target)
    excluded = set(ID_COLUMNS + list(targets) + ["period"])
    return [col for col in df.columns if col not in excluded]

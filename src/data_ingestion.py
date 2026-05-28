from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import RAW_DATA_DIR, REAL_DATA_DIR
from config.diseases import DISEASE_TASKS
from utils.io import ensure_dirs


@dataclass(frozen=True)
class IngestionBundle:
    patients: pd.DataFrame
    encounters: pd.DataFrame
    observations: pd.DataFrame
    conditions: pd.DataFrame


OBSERVATION_MAP = {
    "8302-2": "height_cm",
    "29463-7": "weight_kg",
    "8462-4": "diastolic_bp",
    "8480-6": "systolic_bp",
    "39156-5": "bmi",
    "4548-4": "hba1c",
    "2339-0": "glucose",
    "2345-7": "glucose",
    "2093-3": "cholesterol",
    "18262-6": "ldl_cholesterol",
    "2085-9": "hdl_cholesterol",
}


def _read_csv(path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", **kwargs)


def _parse_date(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, dayfirst=dayfirst).dt.tz_localize(None)


def _load_relevant_observations() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    usecols = ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "VALUE", "UNITS", "TYPE"]
    for chunk in pd.read_csv(REAL_DATA_DIR / "observations.csv", usecols=usecols, dtype={"CODE": str}, chunksize=250_000):
        chunk = chunk[chunk["CODE"].isin(OBSERVATION_MAP)]
        chunk = chunk[chunk["TYPE"].astype(str).str.lower().eq("numeric")]
        chunk["value"] = pd.to_numeric(chunk["VALUE"], errors="coerce")
        chunk = chunk.dropna(subset=["value"])
        if not chunk.empty:
            chunk["parameter"] = chunk["CODE"].map(OBSERVATION_MAP)
            rows.append(
                chunk.rename(
                    columns={
                        "PATIENT": "patient_id",
                        "ENCOUNTER": "encounter_id",
                        "DATE": "observation_date",
                        "UNITS": "unit",
                    }
                )[["patient_id", "encounter_id", "observation_date", "parameter", "value", "unit"]]
            )
    if not rows:
        return pd.DataFrame(columns=["observation_id", "patient_id", "encounter_id", "observation_date", "parameter", "value", "unit"])
    observations = pd.concat(rows, ignore_index=True)
    observations["observation_date"] = _parse_date(observations["observation_date"])
    observations.insert(0, "observation_id", range(1, len(observations) + 1))
    return observations.drop_duplicates(["patient_id", "encounter_id", "observation_date", "parameter"])


def _derive_disease_labels(task: dict, conditions: pd.DataFrame, observations: pd.DataFrame) -> pd.Series:
    description = conditions["condition_name"].astype(str)
    include = "|".join(task["condition_patterns"])
    diagnosis_mask = description.str.contains(include, case=False, na=False) if include else pd.Series(False, index=conditions.index)
    for pattern in task.get("exclude_patterns", []):
        diagnosis_mask &= ~description.str.contains(pattern, case=False, na=False)
    diagnosis_positive = conditions.loc[diagnosis_mask, "patient_id"]

    lab_mask = pd.Series(False, index=observations.index)
    for parameter, op, threshold in task.get("lab_rules", []):
        if op == ">=":
            lab_mask |= observations["parameter"].eq(parameter) & (observations["value"] >= threshold)
    lab_positive = observations.loc[lab_mask, "patient_id"]
    return pd.Series(pd.concat([diagnosis_positive, lab_positive]).dropna().unique())


def prepare_real_data() -> IngestionBundle:
    if not REAL_DATA_DIR.exists():
        raise FileNotFoundError(f"Expected real CSV data under {REAL_DATA_DIR}")

    raw_patients = _read_csv(REAL_DATA_DIR / "patients.csv")
    raw_encounters = _read_csv(REAL_DATA_DIR / "encounters.csv", dtype={"Id": str, "PATIENT": str})
    raw_conditions = _read_csv(REAL_DATA_DIR / "conditions.csv", usecols=["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"], dtype={"CODE": str})
    observations = _load_relevant_observations()

    encounters = raw_encounters.rename(
        columns={
            "Id": "encounter_id",
            "PATIENT": "patient_id",
            "START": "visit_date",
            "ENCOUNTERCLASS": "visit_type",
        }
    )[["encounter_id", "patient_id", "visit_date", "visit_type"]].copy()
    encounters["visit_date"] = _parse_date(encounters["visit_date"])
    encounters["visit_type"] = encounters["visit_type"].fillna("unknown").astype(str).str.title()

    conditions = raw_conditions.rename(
        columns={
            "PATIENT": "patient_id",
            "ENCOUNTER": "encounter_id",
            "START": "diagnosis_date",
            "CODE": "condition_code",
            "DESCRIPTION": "condition_name",
        }
    ).copy()
    conditions["diagnosis_date"] = _parse_date(conditions["diagnosis_date"], dayfirst=True)
    conditions.insert(0, "condition_id", range(1, len(conditions) + 1))

    first_last = encounters.groupby("patient_id").agg(first_visit=("visit_date", "min"), last_visit=("visit_date", "max")).reset_index()
    patients = raw_patients.rename(
        columns={
            "Id": "patient_id",
            "BIRTHDATE": "birthdate",
            "GENDER": "gender",
            "RACE": "race",
            "ETHNICITY": "ethnicity",
            "INCOME": "income",
        }
    )[["patient_id", "birthdate", "gender", "race", "ethnicity", "income"]].copy()
    patients = patients.merge(first_last, on="patient_id", how="left")
    patients["birthdate"] = _parse_date(patients["birthdate"])
    reference_date = patients["last_visit"].fillna(pd.Timestamp("2026-03-31"))
    patients["age"] = np.floor((reference_date - patients["birthdate"]).dt.days / 365.25).clip(lower=0, upper=110)
    patients["registration_date"] = patients["first_visit"].fillna(patients["birthdate"])
    patients["gender"] = patients["gender"].map({"M": "Male", "F": "Female"}).fillna("Unknown")
    patients["ethnicity"] = patients["ethnicity"].fillna(patients["race"]).fillna("Unknown").astype(str).str.title()
    patients["socioeconomic_risk"] = pd.qcut(pd.to_numeric(patients["income"], errors="coerce").fillna(patients["income"].median()), 3, labels=["High", "Medium", "Low"], duplicates="drop").astype(str)
    for task in DISEASE_TASKS.values():
        positives = set(_derive_disease_labels(task, conditions, observations))
        patients[task["target"]] = patients["patient_id"].isin(positives).astype(int)
    patients = patients[
        ["patient_id", "age", "gender", "ethnicity", "socioeconomic_risk", "registration_date"]
        + [task["target"] for task in DISEASE_TASKS.values()]
    ]

    ensure_dirs(RAW_DATA_DIR)
    patients.to_csv(RAW_DATA_DIR / "patients.csv", index=False)
    encounters.to_csv(RAW_DATA_DIR / "encounters.csv", index=False)
    observations.to_csv(RAW_DATA_DIR / "observations.csv", index=False)
    conditions.to_csv(RAW_DATA_DIR / "conditions.csv", index=False)

    return IngestionBundle(patients=patients, encounters=encounters, observations=observations, conditions=conditions)


if __name__ == "__main__":
    bundle = prepare_real_data()
    print({name: len(getattr(bundle, name)) for name in ("patients", "encounters", "observations", "conditions")})

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features import feature_columns


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.nunique() < 2 or actual.nunique() < 2:
        return 0.0
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    expected_pct = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    actual_pct = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    expected_pct = np.clip(expected_pct, 0.001, None)
    actual_pct = np.clip(actual_pct, 0.001, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def drift_report(historical: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in feature_columns(historical):
        if pd.api.types.is_numeric_dtype(historical[col]):
            h = historical[col]
            c = current[col]
            ks = ks_2samp(h.dropna(), c.dropna())
            psi = population_stability_index(h, c)
            rows.append(
                {
                    "feature": col,
                    "type": "numeric",
                    "historical_mean": float(h.mean(skipna=True)),
                    "current_mean": float(c.mean(skipna=True)),
                    "mean_shift": float(c.mean(skipna=True) - h.mean(skipna=True)),
                    "psi": psi,
                    "ks_pvalue": float(ks.pvalue),
                    "drift_level": "High" if psi >= 0.25 else "Moderate" if psi >= 0.1 else "Low",
                }
            )
    return pd.DataFrame(rows).sort_values(["psi", "mean_shift"], ascending=[False, False])


def target_shift(features: pd.DataFrame, target: str) -> pd.DataFrame:
    return features.groupby("period")[target].agg(["mean", "count"]).reset_index().rename(columns={"mean": "positive_rate"})

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import ASSET_DIR, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.pipeline import run_pipeline


def apply_theme() -> None:
    css_path = ASSET_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def ensure_artifacts() -> None:
    if not (PROCESSED_DATA_DIR / "metrics.csv").exists():
        with st.spinner("Ingesting local EHR CSVs and training models..."):
            run_pipeline(force=False)


@st.cache_data(show_spinner=False)
def load_frame(path: Path) -> pd.DataFrame:
    parse_dates = [col for col in ["index_date", "visit_date", "observation_date", "diagnosis_date", "registration_date"] if col in pd.read_csv(path, nrows=0).columns]
    return pd.read_csv(path, parse_dates=parse_dates)


def data() -> dict[str, pd.DataFrame]:
    ensure_artifacts()
    frames = {
        "features": load_frame(PROCESSED_DATA_DIR / "features.csv"),
        "metrics": load_frame(PROCESSED_DATA_DIR / "metrics.csv"),
        "drift": load_frame(PROCESSED_DATA_DIR / "drift_report.csv"),
        "target_shift": load_frame(PROCESSED_DATA_DIR / "target_shift.csv"),
        "continual": load_frame(PROCESSED_DATA_DIR / "continual_learning.csv"),
        "patients": load_frame(RAW_DATA_DIR / "patients.csv"),
        "encounters": load_frame(RAW_DATA_DIR / "encounters.csv"),
        "observations": load_frame(RAW_DATA_DIR / "observations.csv"),
        "conditions": load_frame(RAW_DATA_DIR / "conditions.csv"),
    }
    return frames


def task_data(slug: str) -> dict[str, pd.DataFrame]:
    ensure_artifacts()
    task_dir = PROCESSED_DATA_DIR / "tasks" / slug
    return {
        "metrics": load_frame(task_dir / "metrics.csv"),
        "drift": load_frame(task_dir / "drift_report.csv"),
        "target_shift": load_frame(task_dir / "target_shift.csv"),
        "continual": load_frame(task_dir / "continual_learning.csv"),
    }


def metric_card(label: str, value: str, foot: str = "") -> None:
    st.markdown(
        f"""
        <div class="glass-card metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, body: str = "") -> None:
    st.markdown(f"### {title}")
    if body:
        st.caption(body)


def model_path(name: str, slug: str = "diabetes") -> Path:
    return MODEL_DIR / slug / f"{name.lower().replace(' ', '_')}.joblib"

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.diseases import DISEASE_TASKS
from config.settings import PROCESSED_DATA_DIR
from dashboard.components import apply_theme, data, metric_card, model_path, section_header, task_data
from src.features import feature_columns
from src.modeling import predict_scores
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="Clinical AI Continual Learning",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

PAGES = [
    "Home",
    "Dataset Overview",
    "EDA & Visualizations",
    "Feature Engineering",
    "Model Training",
    "Model Comparison",
    "Temporal Drift Analysis",
    "Continual Learning",
    "Feature Importance",
    "Final Insights",
]

with st.sidebar:
    st.markdown("## ClinShift AI")
    st.caption("Temporal healthcare ML control center")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    disease = st.selectbox("Disease program", list(DISEASE_TASKS.keys()))
    if st.button("Run automated pipeline"):
        with st.spinner("Rebuilding data, models, drift reports, and PDF..."):
            result = run_pipeline(force=True)
        st.success(f"Pipeline complete. Trained {len(result.get('tasks', []))} disease tasks.")

frames = data()
features = frames["features"]
task = DISEASE_TASKS[disease]
target = task["target"]
slug = task["slug"]
try:
    task_frames = task_data(slug)
    metrics = task_frames["metrics"]
    drift = task_frames["drift"]
    continual = task_frames["continual"]
except FileNotFoundError:
    st.error(f"No trained artifacts found for {disease}. Run the automated pipeline from the sidebar.")
    st.stop()

if "message" in metrics.columns:
    st.warning(f"{disease} was not trained: {metrics['message'].iloc[0]}")
    st.stop()

def latest_best_model() -> str:
    return metrics[metrics["cohort"] == "Historical Test"].sort_values("roc_auc", ascending=False).iloc[0]["model"]


def plot_metric_cards() -> None:
    best = latest_best_model()
    best_hist = metrics[(metrics["model"] == best) & (metrics["cohort"] == "Historical Test")].iloc[0]
    best_cur = metrics[(metrics["model"] == best) & (metrics["cohort"] == "Current Test")].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Patients", f"{features['patient_id'].nunique():,}", "Physical CSV population")
    with c2:
        metric_card("Best Model", best, f"{disease} AUC {best_hist['roc_auc']:.3f}")
    with c3:
        metric_card("Current AUC", f"{best_cur['roc_auc']:.3f}", f"Shift delta {best_cur['auc_degradation']:.3f}")
    with c4:
        metric_card("High Drift Features", str((drift["drift_level"] == "High").sum()), "PSI >= 0.25")


def home() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <span class="status-pill">Automated clinical prediction under temporal shift</span>
          <h1>ClinShift AI {disease} Command Center</h1>
          <p>End-to-end local EHR CSV ingestion, feature engineering, model search, drift surveillance, explainability, and continual learning in one interactive healthcare ML platform.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    plot_metric_cards()
    section_header("Operating Model", "The system trains only on historical records, tests on both historical and current cohorts, then adapts with current training data.")
    flow = pd.DataFrame(
        {
            "stage": ["Relational EHR", "Feature Factory", "Historical Training", "Temporal Evaluation", "Continual Learning"],
            "detail": ["patients, encounters, observations, conditions", "biometrics, cadence, variability, risk flags", "CV-tuned DT, SVM, MLP", "degradation, PSI, KS, target shift", "expanded retraining and current test lift"],
        }
    )
    st.dataframe(flow, use_container_width=True, hide_index=True)
    summary_path = PROCESSED_DATA_DIR / "task_summary.csv"
    if summary_path.exists():
        section_header("Disease Programs", "Separate binary classifiers are trained and monitored per disease.")
        st.dataframe(pd.read_csv(summary_path).round(4), use_container_width=True, hide_index=True)


def dataset_overview() -> None:
    section_header("Dataset Overview", "Canonicalized relational EHR tables plus a patient-level analytical feature matrix.")
    c1, c2, c3, c4 = st.columns(4)
    for col, name in zip([c1, c2, c3, c4], ["patients", "encounters", "observations", "conditions"]):
        with col:
            metric_card(name.title(), f"{len(frames[name]):,}", f"{frames[name].shape[1]} columns")
    table = st.selectbox("Table", ["patients", "encounters", "observations", "conditions", "features"])
    st.dataframe(frames[table].head(500), use_container_width=True)
    st.plotly_chart(px.bar(pd.DataFrame({"table": list(frames.keys()), "rows": [len(v) for v in frames.values()]}), x="table", y="rows", color="table", template="plotly_dark"), use_container_width=True)


def eda() -> None:
    section_header("EDA & Visualizations", "Interactive cohort, outcome, and clinical measurement exploration.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(features, x="age", color=target, nbins=30, barmode="overlay", template="plotly_dark"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(features, names=target, title=f"{disease} Class Distribution", template="plotly_dark", hole=0.45), use_container_width=True)
    numeric = features.select_dtypes(include="number")
    corr_cols = st.multiselect("Correlation features", numeric.columns.tolist(), default=[c for c in ["age", "glucose_mean", "hba1c_mean", "bmi_mean", "systolic_bp_mean", "metabolic_syndrome_score", target] if c in numeric.columns])
    if len(corr_cols) >= 2:
        st.plotly_chart(px.imshow(numeric[corr_cols].corr(), color_continuous_scale="Tealrose", template="plotly_dark", aspect="auto"), use_container_width=True)
    y = st.selectbox("Clinical measure", ["glucose_mean", "hba1c_mean", "bmi_mean", "systolic_bp_mean", "cholesterol_mean"])
    st.plotly_chart(px.box(features, x="period", y=y, color=target, points="outliers", template="plotly_dark"), use_container_width=True)


def feature_engineering() -> None:
    section_header("Feature Engineering", "Aggregated patient-level features designed for chronic disease prediction.")
    engineered = [c for c in features.columns if any(k in c for k in ["mean", "std", "variability", "score", "rate", "visit", "bmi", "flag"])]
    st.dataframe(features[["patient_id", target, "period"] + engineered[:25]].head(300), use_container_width=True)
    selected = st.selectbox("Feature distribution", engineered, index=engineered.index("metabolic_syndrome_score") if "metabolic_syndrome_score" in engineered else 0)
    st.plotly_chart(px.histogram(features, x=selected, color="period", marginal="box", template="plotly_dark", barmode="overlay"), use_container_width=True)


def model_training() -> None:
    section_header("Model Training", "Automated preprocessing and randomized hyperparameter search with cross-validation.")
    st.code("python -m src.pipeline", language="powershell")
    st.dataframe(metrics[["model", "cohort", "accuracy", "precision", "recall", "f1", "roc_auc", "auc_degradation"]].round(3), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(metrics, x="model", y="roc_auc", color="cohort", barmode="group", template="plotly_dark"), use_container_width=True)
    best = latest_best_model()
    path = model_path(best, slug)
    if path.exists():
        st.download_button("Download best historical model", data=path.read_bytes(), file_name=path.name)


def model_comparison() -> None:
    section_header("Model Comparison", "Side-by-side model quality, degradation, and confusion matrix inspection.")
    metric = st.selectbox("Metric", ["roc_auc", "f1", "recall", "precision", "accuracy"])
    st.plotly_chart(px.line(metrics, x="cohort", y=metric, color="model", markers=True, template="plotly_dark"), use_container_width=True)
    model = st.selectbox("Confusion matrix model", metrics["model"].unique())
    cohort = st.selectbox("Cohort", metrics["cohort"].unique())
    row = metrics[(metrics["model"] == model) & (metrics["cohort"] == cohort)].iloc[0]
    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    st.plotly_chart(px.imshow(matrix, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=[f"No {disease}", disease], y=[f"No {disease}", disease], template="plotly_dark"), use_container_width=True)


def temporal_drift() -> None:
    section_header("Temporal Drift Analysis", "PSI, KS tests, target shift, and model performance degradation from historical to current cohorts.")
    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.plotly_chart(px.bar(drift.head(18), x="psi", y="feature", color="drift_level", orientation="h", template="plotly_dark"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(task_frames["target_shift"], x="period", y="positive_rate", color="period", template="plotly_dark"), use_container_width=True)
    st.dataframe(drift.round(4), use_container_width=True, hide_index=True)
    high = drift[drift["drift_level"] == "High"]
    if not high.empty:
        st.warning(f"Drift alert: {len(high)} features exceed the high PSI threshold.")


def continual_learning() -> None:
    section_header("Continual Learning", "Best historical model retrained with current training data and re-evaluated on the current test set.")
    st.plotly_chart(px.bar(continual, x="cohort", y=["roc_auc", "f1", "recall"], barmode="group", template="plotly_dark"), use_container_width=True)
    st.dataframe(continual.round(3), use_container_width=True, hide_index=True)
    lift = continual.iloc[1]["roc_auc"] - continual.iloc[0]["roc_auc"]
    st.metric("Current-test AUC lift after adaptation", f"{lift:+.3f}")


def feature_importance_page() -> None:
    section_header("Feature Importance & XAI", "Model-specific global importance plus patient-level prediction inspection.")
    files = sorted((PROCESSED_DATA_DIR / "tasks" / slug).glob("feature_importance_*.csv"))
    names = [p.stem.replace("feature_importance_", "").replace("_", " ").title().replace("Svm", "SVM") for p in files]
    chosen = st.selectbox("Model", names)
    imp = pd.read_csv(files[names.index(chosen)])
    st.plotly_chart(px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h", template="plotly_dark"), use_container_width=True)
    st.caption("SHAP-compatible architecture: tree importances, neural first-layer magnitudes, and SVM permutation-style sensitivity are exposed as global explanation signals.")

    st.markdown("#### Real-time Prediction Form")
    best = latest_best_model()
    model_file = model_path(best, slug)
    if model_file.exists():
        model = joblib.load(model_file)
        sample = features[feature_columns(features, target)].median(numeric_only=True).to_dict()
        cols = st.columns(4)
        sample["age"] = cols[0].number_input("Age", 18, 95, int(sample.get("age", 52)))
        sample["glucose_mean"] = cols[1].number_input("Avg glucose", 65.0, 260.0, float(sample.get("glucose_mean", 118.0)))
        sample["hba1c_mean"] = cols[2].number_input("HbA1c", 4.0, 12.0, float(sample.get("hba1c_mean", 6.1)))
        sample["bmi_mean"] = cols[3].number_input("BMI", 16.0, 55.0, float(sample.get("bmi_mean", 29.0)))
        base = features[feature_columns(features, target)].iloc[[0]].copy()
        for key, value in sample.items():
            if key in base:
                base.loc[:, key] = value
        proba = predict_scores(model, base)[0]
        st.metric(f"Predicted {disease.lower()} risk", f"{proba:.1%}")

    upload = st.file_uploader("Upload feature CSV for batch scoring", type=["csv"])
    if upload and model_file.exists():
        batch = pd.read_csv(upload)
        needed = feature_columns(features, target)
        for col in needed:
            if col not in batch:
                batch[col] = np.nan
        scores = predict_scores(joblib.load(model_file), batch[needed])
        out = batch.copy()
        out[f"{slug}_risk"] = scores
        st.dataframe(out.head(200), use_container_width=True)


def final_insights() -> None:
    section_header("Final Insights", "Executive summary for portfolio review and future productionization.")
    best = latest_best_model()
    best_current = metrics[(metrics["model"] == best) & (metrics["cohort"] == "Current Test")].iloc[0]
    st.markdown(
        f"""
        <div class="glass-card">
        <h3>Key Takeaways</h3>
        <p>The best historical model is <b>{best}</b>, with current-cohort ROC-AUC of <b>{best_current['roc_auc']:.3f}</b>. Temporal shift is visible through PSI-ranked feature movement and performance degradation, making continual learning clinically relevant for this demo.</p>
        <p>Recommended next steps: connect real EHR extracts, add fairness slices, validate calibration, formalize model cards, and add governance approval gates before deployment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    report = PROCESSED_DATA_DIR / "tasks" / slug / f"{slug}_prediction_report.pdf"
    if report.exists():
        st.download_button("Download PDF report", data=report.read_bytes(), file_name=report.name)


ROUTES = {
    "Home": home,
    "Dataset Overview": dataset_overview,
    "EDA & Visualizations": eda,
    "Feature Engineering": feature_engineering,
    "Model Training": model_training,
    "Model Comparison": model_comparison,
    "Temporal Drift Analysis": temporal_drift,
    "Continual Learning": continual_learning,
    "Feature Importance": feature_importance_page,
    "Final Insights": final_insights,
}

ROUTES[page]()

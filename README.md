# Automated Clinical Prediction & Continual Learning Dashboard

A production-style Streamlit platform for multi-disease prediction under temporal data shift. The project ingests local relational EHR CSV exports, engineers patient-level clinical features, trains separate binary classifiers per disease, evaluates temporal drift, applies continual learning, and presents everything in a polished dark clinical analytics dashboard.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline
streamlit run dashboard/app.py
```

The pipeline is also triggered automatically from the dashboard if artifacts are missing.

## Dataset

The full EHR CSV dataset is hosted externally because several files are too large for normal GitHub storage:

[Download the dataset from Google Drive](https://drive.google.com/drive/folders/1dPkA16Cux6zOCpDz32fLY8V66UNKMtB8)

After downloading, place the CSV files in this local folder structure:

```text
new data/
  DATA/
    patients.csv
    encounters.csv
    observations.csv
    conditions.csv
    medications.csv
    procedures.csv
    ...
```

Then run:

```powershell
python -m src.pipeline
```

## Project Layout

```text
assets/              CSS and static dashboard assets
config/              Central settings
dashboard/           Streamlit app and page modules
data/raw/            Canonicalized relational EHR tables
data/processed/      Feature matrices, splits, metrics, drift reports
models/              Persisted sklearn pipelines
notebooks/           Placeholder for EDA notebooks
src/                 Data, feature, model, drift, and continual learning code
utils/               Shared helpers
```

## What It Demonstrates

- Multi-table healthcare CSV ingestion for diabetes, hypertension, obesity, and heart disease prediction
- Separate task folders under `data/processed/tasks/<disease>/` and `models/<disease>/`
- Time-aware historical/current dataset split
- KNN imputation, scaling, one-hot encoding, and patient-level aggregation
- Decision Tree, SVM, and MLP training with cross-validation search
- Baseline vs temporal-shift model evaluation
- Drift reports, PSI, KS tests, target shift, and degradation metrics
- Continual learning through retraining/fine-tuning on current training data
- Interactive Plotly analytics, model comparison, XAI, drift alerts, prediction form, CSV upload, and model download

## Notes

This is a portfolio/demo platform for local EHR-style CSV analysis. It is not a medical device and must not be used for clinical decision making.

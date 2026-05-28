from __future__ import annotations

import pandas as pd

from config.diseases import DISEASE_TASKS
from config.settings import MODEL_DIR, PROCESSED_DATA_DIR
from src.continual_learning import continual_retrain
from src.data_ingestion import prepare_real_data
from src.drift import drift_report, target_shift
from src.features import build_feature_matrix, make_temporal_splits, save_processed
from src.modeling import evaluate_all, feature_importance, train_models
from src.reporting import build_pdf_report
from utils.io import ensure_dirs


def run_pipeline(force: bool = False) -> dict:
    ensure_dirs(PROCESSED_DATA_DIR, MODEL_DIR)
    tasks_dir = PROCESSED_DATA_DIR / "tasks"
    if not force and (PROCESSED_DATA_DIR / "task_summary.csv").exists() and tasks_dir.exists():
        return {"status": "cached"}

    bundle = prepare_real_data()
    features = build_feature_matrix(
        {
            "patients": bundle.patients,
            "encounters": bundle.encounters,
            "observations": bundle.observations,
            "conditions": bundle.conditions,
        }
    )
    save_processed(features)

    summary_rows = []
    first_complete_slug = None
    for disease_name, task in DISEASE_TASKS.items():
        slug = task["slug"]
        target = task["target"]
        task_dir = tasks_dir / slug
        task_model_dir = MODEL_DIR / slug
        ensure_dirs(task_dir, task_model_dir)
        positives = int(features[target].sum())
        status = "complete"
        reason = ""

        try:
            splits = make_temporal_splits(features, target)
            split_counts = {name: split[target].value_counts().to_dict() for name, split in splits.items()}
            if positives < 10 or any(len(counts) < 2 for counts in split_counts.values()):
                status = "skipped"
                reason = "Not enough positive/negative cases across temporal splits."
                pd.DataFrame({"message": [reason]}).to_csv(task_dir / "metrics.csv", index=False)
            else:
                save_processed(features, splits, task_dir)
                models = train_models(splits["historical_train"], target, task_model_dir)
                metrics = evaluate_all(models, splits["historical_test"], splits["current_test"], splits["historical_train"], target)
                metrics.insert(0, "disease", disease_name)
                metrics.to_csv(task_dir / "metrics.csv", index=False)

                best_row = metrics[metrics["cohort"] == "Historical Test"].sort_values("roc_auc", ascending=False).iloc[0]
                best_name = best_row["model"]
                _, cl_metrics = continual_retrain(
                    best_name,
                    models[best_name],
                    splits["historical_train"],
                    splits["current_train"],
                    splits["current_test"],
                    target,
                    task_model_dir,
                )
                cl_metrics.insert(0, "disease", disease_name)
                cl_metrics.to_csv(task_dir / "continual_learning.csv", index=False)

                drift = drift_report(splits["historical_train"], splits["current_test"])
                drift.to_csv(task_dir / "drift_report.csv", index=False)
                target_shift(features, target).to_csv(task_dir / "target_shift.csv", index=False)

                for name, model in models.items():
                    feature_importance(model, splits["historical_train"], target).assign(model=name).to_csv(
                        task_dir / f"feature_importance_{name.lower().replace(' ', '_')}.csv",
                        index=False,
                    )
                report_path = build_pdf_report(metrics, drift, task_dir / f"{slug}_prediction_report.pdf")
                if first_complete_slug is None:
                    first_complete_slug = slug
                    metrics.to_csv(PROCESSED_DATA_DIR / "metrics.csv", index=False)
                    cl_metrics.to_csv(PROCESSED_DATA_DIR / "continual_learning.csv", index=False)
                    drift.to_csv(PROCESSED_DATA_DIR / "drift_report.csv", index=False)
                    target_shift(features, target).to_csv(PROCESSED_DATA_DIR / "target_shift.csv", index=False)
            summary_rows.append(
                {
                    "disease": disease_name,
                    "slug": slug,
                    "target": target,
                    "status": status,
                    "reason": reason,
                    "patients": len(features),
                    "positives": positives,
                    "positive_rate": positives / max(len(features), 1),
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "disease": disease_name,
                    "slug": slug,
                    "target": target,
                    "status": "failed",
                    "reason": str(exc),
                    "patients": len(features),
                    "positives": positives,
                    "positive_rate": positives / max(len(features), 1),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PROCESSED_DATA_DIR / "task_summary.csv", index=False)
    return {"status": "complete", "tasks": summary_rows, "rows": len(features)}


if __name__ == "__main__":
    print(run_pipeline(force=True))

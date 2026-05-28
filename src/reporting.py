from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_pdf_report(metrics: pd.DataFrame, drift: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ModuleNotFoundError:
        html_path = output_path.with_suffix(".html")
        html_path.write_text(
            "<h1>Clinical Prediction & Continual Learning Report</h1>"
            "<p>Local EHR CSV clinical prediction demo. Not for clinical use.</p>"
            "<h2>Model Performance</h2>"
            + metrics[["model", "cohort", "roc_auc", "f1", "recall", "precision"]].round(3).to_html(index=False)
            + "<h2>Top Drifted Features</h2>"
            + drift[["feature", "psi", "drift_level"]].round(3).head(10).to_html(index=False),
            encoding="utf-8",
        )
        return html_path

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Clinical Prediction & Continual Learning Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Local EHR CSV clinical prediction demo. Not for clinical use.", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Model Performance", styles["Heading2"]),
    ]
    perf = metrics[["model", "cohort", "roc_auc", "f1", "recall", "precision"]].round(3).head(12)
    table = Table([perf.columns.tolist()] + perf.astype(str).values.tolist())
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243b55")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    story.extend([table, Spacer(1, 12), Paragraph("Top Drifted Features", styles["Heading2"])])
    drift_small = drift[["feature", "psi", "drift_level"]].round(3).head(10)
    story.append(Table([drift_small.columns.tolist()] + drift_small.astype(str).values.tolist()))
    doc.build(story)
    return output_path

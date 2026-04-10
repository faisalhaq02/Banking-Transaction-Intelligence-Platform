from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/opt/project")
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_METRICS_FILE = BASE_DIR / "models" / "customer_segmentation_metrics.json"
ANOMALY_SUMMARY_FILE = OUTPUT_DIR / "anomaly_summary.json"
RISK_SUMMARY_FILE = OUTPUT_DIR / "risk_bucket_summary.json"
SPEND_METRICS_FILE = OUTPUT_DIR / "spend_prediction_metrics.json"

SEGMENT_DATA_FILE = OUTPUT_DIR / "customer_segments.parquet"
RISK_DATA_FILE = OUTPUT_DIR / "customer_risk_scores.parquet"

MONITORING_JSON = OUTPUT_DIR / "model_monitoring.json"
RUN_HISTORY_CSV = OUTPUT_DIR / "ml_run_history.csv"


def safe_read_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def main() -> None:
    print("===== Model Monitoring Started =====")

    segmentation_metrics = safe_read_json(SEGMENT_METRICS_FILE)
    anomaly_summary = safe_read_json(ANOMALY_SUMMARY_FILE)
    risk_summary = safe_read_json(RISK_SUMMARY_FILE)
    spend_metrics = safe_read_json(SPEND_METRICS_FILE)

    segment_distribution = {}
    if SEGMENT_DATA_FILE.exists():
        seg_df = pd.read_parquet(SEGMENT_DATA_FILE).drop_duplicates(subset=["customer_id"]).copy()
        if "segment_name" in seg_df.columns:
            segment_distribution = {
                k: int(v) for k, v in seg_df["segment_name"].value_counts().to_dict().items()
            }

    avg_risk_score = None
    if RISK_DATA_FILE.exists():
        risk_df = pd.read_parquet(RISK_DATA_FILE).drop_duplicates(subset=["customer_id"]).copy()
        if "risk_score" in risk_df.columns:
            avg_risk_score = float(risk_df["risk_score"].mean())

    run_record = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_customers_scored": anomaly_summary.get("total_customers_scored"),
        "best_k": segmentation_metrics.get("best_k"),
        "segmentation_silhouette_score": segmentation_metrics.get("selected_silhouette_score"),
        "segment_distribution": segment_distribution,
        "ensemble_anomaly_rate": anomaly_summary.get("ensemble_anomaly_rate"),
        "ensemble_anomalies": anomaly_summary.get("ensemble_anomalies"),
        "severity_distribution": anomaly_summary.get("severity_distribution"),
        "average_risk_score": avg_risk_score,
        "high_risk_rate": risk_summary.get("high_risk_rate"),
        "risk_bucket_distribution": risk_summary.get("risk_bucket_distribution"),
        "spend_prediction_mae": spend_metrics.get("mae"),
        "spend_prediction_rmse": spend_metrics.get("rmse"),
        "spend_prediction_r2": spend_metrics.get("r2"),
        "spend_model_type": spend_metrics.get("model_type"),
    }

    with open(MONITORING_JSON, "w") as f:
        json.dump(run_record, f, indent=2)

    current_row = pd.DataFrame([run_record])

    if RUN_HISTORY_CSV.exists():
        history_df = pd.read_csv(RUN_HISTORY_CSV)
        history_df = pd.concat([history_df, current_row], ignore_index=True)
    else:
        history_df = current_row

    history_df.to_csv(RUN_HISTORY_CSV, index=False)

    print("\n===== Model Monitoring Complete =====")
    print(f"Saved monitoring snapshot to: {MONITORING_JSON}")
    print(f"Saved run history to: {RUN_HISTORY_CSV}")
    print("\n===== Current Run Record =====")
    print(json.dumps(run_record, indent=2))


if __name__ == "__main__":
    main()
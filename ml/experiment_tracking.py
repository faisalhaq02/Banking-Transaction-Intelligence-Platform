from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/opt/project")
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_REGISTRY_FILE = OUTPUT_DIR / "model_registry.json"
MODEL_MONITORING_FILE = OUTPUT_DIR / "model_monitoring.json"
DRIFT_REPORT_FILE = OUTPUT_DIR / "data_drift_report.json"
RETRAINING_DECISION_FILE = OUTPUT_DIR / "retraining_decision.json"
MODEL_PROMOTION_FILE = OUTPUT_DIR / "model_promotion_decision.json"

EXPERIMENT_SNAPSHOT_FILE = OUTPUT_DIR / "experiment_tracking.json"
EXPERIMENT_HISTORY_FILE = OUTPUT_DIR / "experiment_history.csv"


def safe_read_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def main() -> None:
    print("===== Experiment Tracking Started =====")

    registry = safe_read_json(MODEL_REGISTRY_FILE)
    monitoring = safe_read_json(MODEL_MONITORING_FILE)
    drift = safe_read_json(DRIFT_REPORT_FILE)
    retraining = safe_read_json(RETRAINING_DECISION_FILE)
    promotion = safe_read_json(MODEL_PROMOTION_FILE)

    spend_registry = registry.get("models", {}).get("spend_prediction", {})
    seg_registry = registry.get("models", {}).get("customer_segmentation", {})
    anomaly_registry = registry.get("models", {}).get("customer_anomaly_detection", {})

    experiment_record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_timestamp_utc": monitoring.get("run_timestamp_utc"),
        "num_customers_scored": monitoring.get("num_customers_scored"),
        "segmentation_latest_version": seg_registry.get("latest_version"),
        "segmentation_production_version": seg_registry.get("production_version"),
        "spend_latest_version": spend_registry.get("latest_version"),
        "spend_production_version": spend_registry.get("production_version"),
        "anomaly_latest_version": anomaly_registry.get("latest_version"),
        "anomaly_production_version": anomaly_registry.get("production_version"),
        "segmentation_silhouette_score": monitoring.get("segmentation_silhouette_score"),
        "ensemble_anomaly_rate": monitoring.get("ensemble_anomaly_rate"),
        "high_risk_rate": monitoring.get("high_risk_rate"),
        "average_risk_score": monitoring.get("average_risk_score"),
        "spend_prediction_mae": monitoring.get("spend_prediction_mae"),
        "spend_prediction_rmse": monitoring.get("spend_prediction_rmse"),
        "spend_prediction_r2": monitoring.get("spend_prediction_r2"),
        "drift_status": drift.get("status"),
        "retraining_required": retraining.get("should_retrain"),
        "promotion_performed": promotion.get("should_promote"),
        "promotion_model_name": promotion.get("model_name"),
        "promotion_from_version": promotion.get("production_version_before_decision"),
        "promotion_to_version": promotion.get("production_version_after_decision"),
    }

    with open(EXPERIMENT_SNAPSHOT_FILE, "w") as f:
        json.dump(experiment_record, f, indent=2)

    current_row = pd.DataFrame([experiment_record])

    if EXPERIMENT_HISTORY_FILE.exists():
        history_df = pd.read_csv(EXPERIMENT_HISTORY_FILE)
        history_df = pd.concat([history_df, current_row], ignore_index=True)
    else:
        history_df = current_row

    history_df.to_csv(EXPERIMENT_HISTORY_FILE, index=False)

    print("\n===== Experiment Tracking Complete =====")
    print(f"Saved snapshot to: {EXPERIMENT_SNAPSHOT_FILE}")
    print(f"Saved history to: {EXPERIMENT_HISTORY_FILE}")
    print(json.dumps(experiment_record, indent=2))


if __name__ == "__main__":
    main()
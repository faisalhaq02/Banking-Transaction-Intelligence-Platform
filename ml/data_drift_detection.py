from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/opt/project")
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_HISTORY_FILE = OUTPUT_DIR / "ml_run_history.csv"
CURRENT_MONITORING_FILE = OUTPUT_DIR / "model_monitoring.json"
DRIFT_REPORT_FILE = OUTPUT_DIR / "data_drift_report.json"


def safe_read_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    if previous == 0:
        return None
    return (current - previous) / previous


def drift_level(abs_change: float | None, low: float, medium: float) -> str:
    if abs_change is None:
        return "unknown"
    if abs_change < low:
        return "stable"
    if abs_change < medium:
        return "moderate"
    return "high"


def normalize_distribution(value) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    if isinstance(value, dict):
        total = sum(float(v) for v in value.values()) or 1.0
        return {str(k): float(v) / total for k, v in value.items()}
    return {}


def distribution_drift(curr: dict[str, float], prev: dict[str, float]) -> dict:
    keys = sorted(set(curr.keys()) | set(prev.keys()))
    deltas = {}
    max_abs_delta = 0.0
    for k in keys:
        c = float(curr.get(k, 0.0))
        p = float(prev.get(k, 0.0))
        d = c - p
        deltas[k] = {
            "current_share": c,
            "previous_share": p,
            "delta": d,
        }
        max_abs_delta = max(max_abs_delta, abs(d))
    return {
        "details": deltas,
        "max_absolute_share_change": max_abs_delta,
        "level": drift_level(max_abs_delta, low=0.02, medium=0.05),
    }


def main() -> None:
    print("===== Data Drift Detection Started =====")

    current = safe_read_json(CURRENT_MONITORING_FILE)
    if not current:
        raise FileNotFoundError(f"Missing current monitoring file: {CURRENT_MONITORING_FILE}")

    if not RUN_HISTORY_FILE.exists():
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_history",
            "message": "No run history found yet. Need at least 2 runs for drift comparison.",
        }
        with open(DRIFT_REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    history = pd.read_csv(RUN_HISTORY_FILE)
    if len(history) < 2:
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_history",
            "message": "Only one run found. Need at least 2 runs for drift comparison.",
        }
        with open(DRIFT_REPORT_FILE, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return

    previous_row = history.iloc[-2].to_dict()

    current_segment_dist = normalize_distribution(current.get("segment_distribution"))
    previous_segment_dist = normalize_distribution(previous_row.get("segment_distribution"))

    current_risk_dist = normalize_distribution(current.get("risk_bucket_distribution"))
    previous_risk_dist = normalize_distribution(previous_row.get("risk_bucket_distribution"))

    current_severity_dist = normalize_distribution(current.get("severity_distribution"))
    previous_severity_dist = normalize_distribution(previous_row.get("severity_distribution"))

    metric_changes = {
        "ensemble_anomaly_rate": {
            "current": current.get("ensemble_anomaly_rate"),
            "previous": previous_row.get("ensemble_anomaly_rate"),
        },
        "high_risk_rate": {
            "current": current.get("high_risk_rate"),
            "previous": previous_row.get("high_risk_rate"),
        },
        "average_risk_score": {
            "current": current.get("average_risk_score"),
            "previous": previous_row.get("average_risk_score"),
        },
        "spend_prediction_mae": {
            "current": current.get("spend_prediction_mae"),
            "previous": previous_row.get("spend_prediction_mae"),
        },
        "spend_prediction_rmse": {
            "current": current.get("spend_prediction_rmse"),
            "previous": previous_row.get("spend_prediction_rmse"),
        },
        "spend_prediction_r2": {
            "current": current.get("spend_prediction_r2"),
            "previous": previous_row.get("spend_prediction_r2"),
        },
    }

    for metric_name, info in metric_changes.items():
        curr = None if pd.isna(info["current"]) else float(info["current"])
        prev = None if pd.isna(info["previous"]) else float(info["previous"])
        change = pct_change(curr, prev)
        info["relative_change"] = change
        info["absolute_change"] = None if curr is None or prev is None else curr - prev

        if metric_name in {"spend_prediction_mae", "spend_prediction_rmse"}:
            # lower is better
            magnitude = abs(change) if change is not None else None
        else:
            magnitude = abs(change) if change is not None else None

        info["level"] = drift_level(magnitude, low=0.05, medium=0.15)

    segment_drift = distribution_drift(current_segment_dist, previous_segment_dist)
    risk_bucket_drift = distribution_drift(current_risk_dist, previous_risk_dist)
    severity_drift = distribution_drift(current_severity_dist, previous_severity_dist)

    overall_flags = {
        "segment_distribution_drift": segment_drift["level"],
        "risk_bucket_drift": risk_bucket_drift["level"],
        "severity_distribution_drift": severity_drift["level"],
        "anomaly_rate_drift": metric_changes["ensemble_anomaly_rate"]["level"],
        "high_risk_rate_drift": metric_changes["high_risk_rate"]["level"],
        "average_risk_score_drift": metric_changes["average_risk_score"]["level"],
        "spend_mae_drift": metric_changes["spend_prediction_mae"]["level"],
        "spend_rmse_drift": metric_changes["spend_prediction_rmse"]["level"],
        "spend_r2_drift": metric_changes["spend_prediction_r2"]["level"],
    }

    high_count = sum(1 for v in overall_flags.values() if v == "high")
    moderate_count = sum(1 for v in overall_flags.values() if v == "moderate")

    overall_status = "stable"
    if high_count >= 1:
        overall_status = "high_drift_detected"
    elif moderate_count >= 2:
        overall_status = "moderate_drift_detected"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "current_run_timestamp_utc": current.get("run_timestamp_utc"),
        "previous_run_timestamp_utc": previous_row.get("run_timestamp_utc"),
        "metric_changes": metric_changes,
        "segment_distribution_drift": segment_drift,
        "risk_bucket_drift": risk_bucket_drift,
        "severity_distribution_drift": severity_drift,
        "overall_flags": overall_flags,
        "summary": {
            "high_flags": high_count,
            "moderate_flags": moderate_count,
        },
    }

    with open(DRIFT_REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== Data Drift Detection Complete =====")
    print(f"Saved drift report to: {DRIFT_REPORT_FILE}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
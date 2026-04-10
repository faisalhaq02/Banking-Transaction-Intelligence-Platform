from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path("/opt/project")
OUTPUT_DIR = BASE_DIR / "outputs"

DRIFT_REPORT_FILE = OUTPUT_DIR / "data_drift_report.json"
SPEND_METRICS_FILE = OUTPUT_DIR / "spend_prediction_metrics.json"
DECISION_FILE = OUTPUT_DIR / "retraining_decision.json"


def safe_read_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def main() -> None:
    print("===== Retraining Decision Started =====")

    drift_report = safe_read_json(DRIFT_REPORT_FILE)
    spend_metrics = safe_read_json(SPEND_METRICS_FILE)

    drift_status = drift_report.get("status", "unknown")
    r2 = spend_metrics.get("r2")

    reasons = []
    should_retrain = False

    if drift_status == "high_drift_detected":
        should_retrain = True
        reasons.append("High drift detected in latest monitoring report.")

    if r2 is not None and float(r2) < 0.95:
        should_retrain = True
        reasons.append(f"Spend prediction R2 below threshold: {r2:.4f} < 0.95")

    if not reasons:
        reasons.append("No retraining trigger conditions met.")

    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "should_retrain": should_retrain,
        "drift_status": drift_status,
        "spend_prediction_r2": r2,
        "rules": {
            "high_drift_detected": True,
            "min_spend_prediction_r2": 0.95,
        },
        "reasons": reasons,
    }

    with open(DECISION_FILE, "w") as f:
        json.dump(decision, f, indent=2)

    print("\n===== Retraining Decision Complete =====")
    print(f"Saved decision to: {DECISION_FILE}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
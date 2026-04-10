# ml/enrich_risk_outputs.py

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("/opt/project/outputs/customer_risk_scores.parquet")
OUTPUT_PATH = Path("/opt/project/outputs/customer_risk_scores.parquet")
CSV_OUTPUT_PATH = Path("/opt/project/outputs/customer_risk_scores_enriched.csv")


def map_fraud_risk_level(risk_bucket: str) -> str:
    mapping = {
        "Low Risk": "Low Fraud Risk",
        "Medium Risk": "Moderate Fraud Risk",
        "High Risk": "High Fraud Risk",
    }
    return mapping.get(str(risk_bucket).strip(), "Unknown Fraud Risk")


def build_driver_candidates(row: pd.Series) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []

    candidates.append(("High Transaction Count", float(row.get("txn_count_score", 0.0))))
    candidates.append(("High Total Spend", float(row.get("total_spend_score", 0.0))))
    candidates.append(("High Average Transaction", float(row.get("avg_amount_score", 0.0))))
    candidates.append(("High Maximum Transaction", float(row.get("max_amount_score", 0.0))))
    candidates.append(("High Transaction Intensity", float(row.get("txn_intensity_score", 0.0))))
    candidates.append(("Strong Anomaly Vote", float(row.get("anomaly_vote_score", 0.0))))
    candidates.append(("Strong Anomaly Strength", float(row.get("anomaly_strength_score", 0.0))))
    candidates.append(("Segment-Based Risk", float(row.get("segment_risk_score", 0.0))))

    return sorted(candidates, key=lambda x: x[1], reverse=True)


def build_fraud_reason(row: pd.Series, top_drivers: list[str]) -> str:
    reasons: list[str] = []

    risk_bucket = str(row.get("risk_bucket", ""))
    anomaly_flag = int(row.get("ensemble_anomaly_flag", 0))
    anomaly_votes = int(row.get("ensemble_anomaly_votes", 0))
    anomaly_severity = str(row.get("anomaly_severity", ""))
    segment_name = str(row.get("segment_name", ""))

    if anomaly_flag == 1:
        reasons.append(
            f"customer shows {anomaly_severity.lower() or 'unusual'} behavior with {anomaly_votes} ensemble anomaly vote(s)"
        )

    if top_drivers:
        reasons.append("top driver: " + top_drivers[0].lower())

    if len(top_drivers) > 1:
        reasons.append("secondary driver: " + top_drivers[1].lower())

    if segment_name:
        reasons.append(f"customer belongs to segment '{segment_name}'")

    if not reasons:
        reasons.append(f"customer classified as {risk_bucket.lower()} based on aggregated behavioral features")

    return "; ".join(reasons)


def enrich_risk_file(input_path: Path, output_path: Path, csv_output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path).copy()

    # Business-facing label
    df["fraud_risk_level"] = df["risk_bucket"].apply(map_fraud_risk_level)

    # Timestamp showing when the ML enrichment ran
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    df["model_run_timestamp"] = run_ts

    # Driver columns
    top1: list[str] = []
    top2: list[str] = []
    top3: list[str] = []
    fraud_reason: list[str] = []

    for _, row in df.iterrows():
        ranked = build_driver_candidates(row)
        drivers = [name for name, score in ranked if score > 0]

        d1 = drivers[0] if len(drivers) > 0 else "No Strong Driver"
        d2 = drivers[1] if len(drivers) > 1 else "No Secondary Driver"
        d3 = drivers[2] if len(drivers) > 2 else "No Third Driver"

        top1.append(d1)
        top2.append(d2)
        top3.append(d3)
        fraud_reason.append(build_fraud_reason(row, [d1, d2, d3]))

    df["top_risk_driver_1"] = top1
    df["top_risk_driver_2"] = top2
    df["top_risk_driver_3"] = top3
    df["fraud_reason"] = fraud_reason

    # Optional simple flag for easier BI filtering
    df["is_high_fraud_risk"] = np.where(df["fraud_risk_level"] == "High Fraud Risk", 1, 0)

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    df.to_csv(csv_output_path, index=False)

    print("=" * 90)
    print(f"Enriched parquet written to: {output_path}")
    print(f"Enriched csv written to:     {csv_output_path}")
    print("New columns added:")
    print([
        "fraud_risk_level",
        "model_run_timestamp",
        "top_risk_driver_1",
        "top_risk_driver_2",
        "top_risk_driver_3",
        "fraud_reason",
        "is_high_fraud_risk",
    ])
    print("\nSample rows:")
    cols_to_show = [
        "customer_id",
        "risk_score",
        "risk_bucket",
        "fraud_risk_level",
        "top_risk_driver_1",
        "top_risk_driver_2",
        "top_risk_driver_3",
        "fraud_reason",
        "model_run_timestamp",
    ]
    existing = [c for c in cols_to_show if c in df.columns]
    print(df[existing].head(5).to_string(index=False))


if __name__ == "__main__":
    enrich_risk_file(INPUT_PATH, OUTPUT_PATH, CSV_OUTPUT_PATH)
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("/opt/project")
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    OUTPUT_DIR / "customer_anomalies.parquet",
    OUTPUT_DIR / "customer_segments.parquet",
]

RISK_OUTPUT_FILE = OUTPUT_DIR / "customer_risk_scores.parquet"
TOP_RISK_FILE = OUTPUT_DIR / "top_high_risk_customers.csv"
RISK_SUMMARY_FILE = OUTPUT_DIR / "risk_bucket_summary.json"

BASE_FEATURES = [
    "txn_count",
    "total_spend",
    "avg_amount",
    "max_amount",
    "txn_intensity",
]


def load_input_data() -> pd.DataFrame:
    for path in INPUT_CANDIDATES:
        if path.exists():
            print(f"Loading input data from: {path}")
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
    raise FileNotFoundError(f"Could not find input data in: {INPUT_CANDIDATES}")


def minmax_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(series.median())
    min_v = s.min()
    max_v = s.max()
    if max_v == min_v:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v)


def bucket_from_score(score: float) -> str:
    if score < 0.35:
        return "Low Risk"
    if score < 0.65:
        return "Medium Risk"
    return "High Risk"


def main() -> None:
    print("===== Customer Risk Scoring Started =====")

    df = load_input_data().copy()
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if "customer_id" not in df.columns:
        raise ValueError("customer_id column is required")

    # Ensure one row per customer
    df = df.drop_duplicates(subset=["customer_id"]).copy()

    # Rebuild txn_intensity if needed
    if "txn_intensity" not in df.columns:
        if {"txn_count", "avg_amount"}.issubset(df.columns):
            df["txn_intensity"] = df["txn_count"] * df["avg_amount"]
        else:
            raise ValueError("txn_intensity missing and cannot be derived")

    for col in BASE_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Base behavioral scores
    df["txn_count_score"] = minmax_score(df["txn_count"])
    df["total_spend_score"] = minmax_score(df["total_spend"])
    df["avg_amount_score"] = minmax_score(df["avg_amount"])
    df["max_amount_score"] = minmax_score(df["max_amount"])
    df["txn_intensity_score"] = minmax_score(df["txn_intensity"])

    # Anomaly component
    if "ensemble_anomaly_votes" in df.columns:
        df["anomaly_vote_score"] = df["ensemble_anomaly_votes"] / 3.0
    else:
        df["anomaly_vote_score"] = 0.0

    if "ensemble_anomaly_score" in df.columns:
        df["anomaly_strength_score"] = pd.to_numeric(
            df["ensemble_anomaly_score"], errors="coerce"
        ).fillna(0.0)
    else:
        df["anomaly_strength_score"] = 0.0

    # Segment-based uplift
    segment_risk_map = {}
    if "segment_name" in df.columns:
        segment_risk_map = {
            "Low Activity Customers": 0.10,
            "Regular Customers": 0.25,
            "High Value Customers": 0.55,
            "Extreme Spenders": 0.80,
            "Elite Customers": 0.85,
            "Premium Customers": 0.70,
        }
        df["segment_risk_score"] = df["segment_name"].map(segment_risk_map).fillna(0.30)
    else:
        df["segment_risk_score"] = 0.30

    # Final weighted risk score
    df["risk_score"] = (
        0.12 * df["txn_count_score"] +
        0.22 * df["total_spend_score"] +
        0.10 * df["avg_amount_score"] +
        0.16 * df["max_amount_score"] +
        0.12 * df["txn_intensity_score"] +
        0.12 * df["anomaly_vote_score"] +
        0.08 * df["anomaly_strength_score"] +
        0.08 * df["segment_risk_score"]
    )

    df["risk_score"] = df["risk_score"].clip(0, 1)
    df["risk_bucket"] = df["risk_score"].apply(bucket_from_score)

    df = df.sort_values(["risk_score", "total_spend"], ascending=[False, False]).reset_index(drop=True)

    top_cols = [
        "customer_id",
        "risk_score",
        "risk_bucket",
        "txn_count",
        "total_spend",
        "avg_amount",
        "max_amount",
    ]

    optional_cols = [
        "customer_segment",
        "segment_name",
        "ensemble_anomaly_votes",
        "ensemble_anomaly_flag",
        "anomaly_severity",
        "ensemble_anomaly_score",
    ]
    for col in optional_cols:
        if col in df.columns:
            top_cols.append(col)

    top_high_risk = df[top_cols].head(100).copy()

    summary = {
        "total_customers_scored": int(len(df)),
        "risk_bucket_distribution": {
            k: int(v) for k, v in df["risk_bucket"].value_counts().to_dict().items()
        },
        "average_risk_score": float(df["risk_score"].mean()),
        "high_risk_rate": float((df["risk_bucket"] == "High Risk").mean()),
        "score_weights": {
            "txn_count_score": 0.12,
            "total_spend_score": 0.22,
            "avg_amount_score": 0.10,
            "max_amount_score": 0.16,
            "txn_intensity_score": 0.12,
            "anomaly_vote_score": 0.12,
            "anomaly_strength_score": 0.08,
            "segment_risk_score": 0.08,
        },
        "segment_risk_map": segment_risk_map,
    }

    df.to_parquet(RISK_OUTPUT_FILE, index=False)
    top_high_risk.to_csv(TOP_RISK_FILE, index=False)

    with open(RISK_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Customer Risk Scoring Complete =====")
    print(f"Saved risk output to: {RISK_OUTPUT_FILE}")
    print(f"Saved top risk customers to: {TOP_RISK_FILE}")
    print(f"Saved risk summary to: {RISK_SUMMARY_FILE}")

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2))

    print("\nTop high risk preview:")
    print(top_high_risk.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
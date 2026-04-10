from __future__ import annotations

import os
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path("/opt/project")
DATA_DIR = BASE_DIR / "data"
GOLD_DIR = DATA_DIR / "gold"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    GOLD_DIR / "customer_features.parquet",
    GOLD_DIR / "customer_features.csv",
    GOLD_DIR / "customer_features",
]

SEGMENT_OUTPUT_FILE = OUTPUT_DIR / "customer_segments.parquet"
SEGMENT_PROFILE_FILE = OUTPUT_DIR / "segment_profiles.csv"
MODEL_FILE = MODEL_DIR / "customer_segmentation_kmeans.pkl"
SCALER_FILE = MODEL_DIR / "customer_segmentation_scaler.pkl"
METRICS_FILE = MODEL_DIR / "customer_segmentation_metrics.json"


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def load_input_data() -> pd.DataFrame:
    for path in INPUT_CANDIDATES:
        if path.exists():
            print(f"Loading input data from: {path}")
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
            # fallback if path exists without suffix
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.read_csv(path)

    raise FileNotFoundError(
        f"Could not find customer features input in any of these locations: {INPUT_CANDIDATES}"
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    df = df.copy()

    required_cols = ["txn_count", "total_spend", "avg_amount", "max_amount"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Detect customer id column
    customer_id_col = "customer_id" if "customer_id" in df.columns else None
    if customer_id_col is None:
        df["customer_id"] = np.arange(1, len(df) + 1)
        customer_id_col = "customer_id"

    # Fill nulls safely
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Extra engineered features for better segmentation
    df["spend_per_txn_ratio"] = np.where(
        df["txn_count"] > 0,
        df["total_spend"] / df["txn_count"],
        0
    )
    df["max_to_avg_ratio"] = np.where(
        df["avg_amount"] > 0,
        df["max_amount"] / df["avg_amount"],
        0
    )
    df["amount_range"] = df["max_amount"] - df["avg_amount"]
    df["txn_intensity"] = df["txn_count"] * df["avg_amount"]

    feature_cols = [
        "txn_count",
        "total_spend",
        "avg_amount",
        "max_amount",
        "spend_per_txn_ratio",
        "max_to_avg_ratio",
        "amount_range",
        "txn_intensity",
    ]

    return df, feature_cols, customer_id_col


def find_best_k(X_scaled: np.ndarray, min_k: int = 2, max_k: int = 6) -> tuple[int, dict]:
    scores = {}

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        # silhouette needs at least 2 clusters and less than total samples
        if len(set(labels)) > 1 and len(set(labels)) < len(X_scaled):
            score = silhouette_score(X_scaled, labels)
            scores[k] = float(score)
        else:
            scores[k] = -1.0

    best_k = max(scores, key=scores.get)
    return best_k, scores


def name_segments(profile_df: pd.DataFrame) -> dict:
    """
    Assign human-readable names based on total_spend ranking.
    """
    ranked = profile_df.sort_values("total_spend_mean").reset_index(drop=True)
    segment_names = {}

    labels_pool = [
        "Low Activity Customers",
        "Regular Customers",
        "High Value Customers",
        "Extreme Spenders",
        "Elite Customers",
        "Premium Customers",
    ]

    for i, row in ranked.iterrows():
        segment = int(row["customer_segment"])
        segment_names[segment] = labels_pool[i] if i < len(labels_pool) else f"Segment {segment}"

    return segment_names


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def main() -> None:
    print("===== Customer Segmentation Training Started =====")

    # Load
    df = load_input_data()
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Prepare
    df, feature_cols, customer_id_col = prepare_features(df)
    X = df[feature_cols].copy()

    print(f"Prepared feature shape: {X.shape}")
    print(f"Feature columns: {feature_cols}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose best k
    best_k, silhouette_scores = find_best_k(X_scaled, min_k=2, max_k=6)
    print("\nSilhouette scores by K:")
    for k, score in silhouette_scores.items():
        print(f"K={k}: {score:.4f}")

    print(f"\nSelected best K: {best_k}")

    # Train final model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["customer_segment"] = kmeans.fit_predict(X_scaled)

    # Segment profiles
    profile_df = (
        df.groupby("customer_segment")
        .agg(
            customer_count=(customer_id_col, "count"),
            txn_count_mean=("txn_count", "mean"),
            total_spend_mean=("total_spend", "mean"),
            avg_amount_mean=("avg_amount", "mean"),
            max_amount_mean=("max_amount", "mean"),
            spend_per_txn_ratio_mean=("spend_per_txn_ratio", "mean"),
            txn_intensity_mean=("txn_intensity", "mean"),
        )
        .reset_index()
        .sort_values("total_spend_mean")
    )

    segment_name_map = name_segments(profile_df)
    df["segment_name"] = df["customer_segment"].map(segment_name_map)
    profile_df["segment_name"] = profile_df["customer_segment"].map(segment_name_map)

    # Reorder columns
    output_columns = [customer_id_col] + [c for c in df.columns if c != customer_id_col]
    df = df[output_columns]

    # Save outputs
    df.to_parquet(SEGMENT_OUTPUT_FILE, index=False)
    profile_df.to_csv(SEGMENT_PROFILE_FILE, index=False)

    # Save model artifacts
    joblib.dump(kmeans, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    metrics = {
        "best_k": int(best_k),
        "silhouette_scores": silhouette_scores,
        "selected_silhouette_score": silhouette_scores.get(best_k, None),
        "num_customers": int(len(df)),
        "feature_columns": feature_cols,
        "segment_name_map": {str(k): v for k, v in segment_name_map.items()},
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # Console summary
    print("\n===== Segmentation Complete =====")
    print(f"Saved segmented customers to: {SEGMENT_OUTPUT_FILE}")
    print(f"Saved segment profiles to: {SEGMENT_PROFILE_FILE}")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved scaler to: {SCALER_FILE}")
    print(f"Saved metrics to: {METRICS_FILE}")

    print("\n===== Segment Profiles =====")
    print(profile_df.to_string(index=False))


if __name__ == "__main__":
    main()
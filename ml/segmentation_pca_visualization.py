from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path("/opt/project")
DATA_DIR = BASE_DIR / "data"
GOLD_DIR = DATA_DIR / "gold"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    OUTPUT_DIR / "customer_segments.parquet",
    GOLD_DIR / "customer_features.parquet",
    GOLD_DIR / "customer_features.csv",
]

PCA_PLOT_FILE = OUTPUT_DIR / "pca_customer_segments.png"
PCA_DATA_FILE = OUTPUT_DIR / "customer_segments_pca.parquet"
PCA_METRICS_FILE = OUTPUT_DIR / "pca_explained_variance.json"
PCA_MODEL_FILE = MODEL_DIR / "customer_segmentation_pca.pkl"

FEATURE_COLS = [
    "txn_count",
    "total_spend",
    "avg_amount",
    "max_amount",
    "spend_per_txn_ratio",
    "max_to_avg_ratio",
    "amount_range",
    "txn_intensity",
]


def load_input_data() -> pd.DataFrame:
    for path in INPUT_CANDIDATES:
        if path.exists():
            print(f"Loading input data from: {path}")
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find segmentation input in any of these locations: {INPUT_CANDIDATES}"
    )


def main() -> None:
    print("===== PCA Cluster Visualization Started =====")

    df = load_input_data().copy()
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required PCA feature columns: {missing}")

    if "customer_segment" not in df.columns:
        raise ValueError(
            "Column 'customer_segment' not found. Run train_customer_segmentation.py first."
        )

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLS].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    df["pca_component_1"] = components[:, 0]
    df["pca_component_2"] = components[:, 1]

    explained = pca.explained_variance_ratio_.tolist()
    metrics = {
        "explained_variance_ratio_pc1": float(explained[0]),
        "explained_variance_ratio_pc2": float(explained[1]),
        "total_explained_variance_ratio": float(sum(explained)),
        "n_rows": int(len(df)),
        "feature_columns": FEATURE_COLS,
    }

    # Save PCA data
    df.to_parquet(PCA_DATA_FILE, index=False)
    joblib.dump(pca, PCA_MODEL_FILE)

    with open(PCA_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot
    plt.figure(figsize=(10, 6))

    segments = sorted(df["customer_segment"].dropna().unique())
    for seg in segments:
        subset = df[df["customer_segment"] == seg]
        label_name = (
            subset["segment_name"].iloc[0]
            if "segment_name" in subset.columns and not subset.empty
            else f"Segment {seg}"
        )
        plt.scatter(
            subset["pca_component_1"],
            subset["pca_component_2"],
            label=label_name,
            alpha=0.7,
            s=35,
        )

    plt.xlabel(f"Principal Component 1 ({explained[0]*100:.2f}% variance)")
    plt.ylabel(f"Principal Component 2 ({explained[1]*100:.2f}% variance)")
    plt.title("Customer Segments - PCA Visualization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PCA_PLOT_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n===== PCA Visualization Complete =====")
    print(f"Saved PCA plot to: {PCA_PLOT_FILE}")
    print(f"Saved PCA data to: {PCA_DATA_FILE}")
    print(f"Saved PCA metrics to: {PCA_METRICS_FILE}")
    print(f"Saved PCA model to: {PCA_MODEL_FILE}")
    print(f"Explained variance (PC1 + PC2): {sum(explained):.4f}")


if __name__ == "__main__":
    main()
    
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path("/opt/project")
DATA_DIR = BASE_DIR / "data"
GOLD_DIR = DATA_DIR / "gold"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    OUTPUT_DIR / "customer_segments.parquet",
    GOLD_DIR / "customer_features.parquet",
    GOLD_DIR / "customer_features.csv",
]

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

PROFILE_CSV = OUTPUT_DIR / "segment_feature_profiles.csv"
PROFILE_ZSCORE_CSV = OUTPUT_DIR / "segment_feature_profiles_zscore.csv"
TOP_FEATURES_JSON = OUTPUT_DIR / "segment_top_features.json"
HEATMAP_PNG = OUTPUT_DIR / "segment_feature_heatmap.png"


def load_input_data() -> pd.DataFrame:
    for path in INPUT_CANDIDATES:
        if path.exists():
            print(f"Loading input data from: {path}")
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find input data in: {INPUT_CANDIDATES}")


def main() -> None:
    print("===== Segment Feature Profiles Started =====")

    df = load_input_data().copy()
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if "customer_segment" not in df.columns:
        raise ValueError("Column 'customer_segment' not found. Run train_customer_segmentation.py first.")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Raw means by segment
    profile_df = (
        df.groupby("customer_segment")[FEATURE_COLS]
        .mean()
        .reset_index()
        .sort_values("customer_segment")
    )

    # Attach segment names if present
    if "segment_name" in df.columns:
        name_map = (
            df[["customer_segment", "segment_name"]]
            .drop_duplicates()
            .sort_values("customer_segment")
        )
        profile_df = profile_df.merge(name_map, on="customer_segment", how="left")

    # Z-score standardized version for comparison
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[FEATURE_COLS])
    scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_COLS)
    scaled_df["customer_segment"] = df["customer_segment"].values

    profile_z_df = (
        scaled_df.groupby("customer_segment")[FEATURE_COLS]
        .mean()
        .reset_index()
        .sort_values("customer_segment")
    )

    if "segment_name" in df.columns:
        profile_z_df = profile_z_df.merge(name_map, on="customer_segment", how="left")

    # Top differentiating features per segment
    top_features = {}
    for _, row in profile_z_df.iterrows():
        seg = int(row["customer_segment"])
        row_features = {feature: float(row[feature]) for feature in FEATURE_COLS}
        ranked = sorted(row_features.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features[str(seg)] = {
            "segment_name": row.get("segment_name", f"Segment {seg}"),
            "top_positive_features": ranked[:3],
        }

    # Save outputs
    profile_df.to_csv(PROFILE_CSV, index=False)
    profile_z_df.to_csv(PROFILE_ZSCORE_CSV, index=False)

    with open(TOP_FEATURES_JSON, "w") as f:
        json.dump(top_features, f, indent=2)

    # Heatmap-style plot using matplotlib only
    plot_df = profile_z_df.copy()
    labels = []
    if "segment_name" in plot_df.columns:
        labels = [
            f"{int(r.customer_segment)} - {r.segment_name}"
            for _, r in plot_df.iterrows()
        ]
    else:
        labels = [f"Segment {int(x)}" for x in plot_df["customer_segment"]]

    matrix = plot_df[FEATURE_COLS].values

    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Average Z-Score")
    plt.xticks(range(len(FEATURE_COLS)), FEATURE_COLS, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Segment Feature Heatmap (Z-Score Profiles)")
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n===== Segment Feature Profiles Complete =====")
    print(f"Saved raw profiles to: {PROFILE_CSV}")
    print(f"Saved z-score profiles to: {PROFILE_ZSCORE_CSV}")
    print(f"Saved top features to: {TOP_FEATURES_JSON}")
    print(f"Saved heatmap to: {HEATMAP_PNG}")

    print("\nTop features by segment:")
    for seg, info in top_features.items():
        print(f"Segment {seg} ({info['segment_name']}): {info['top_positive_features']}")


if __name__ == "__main__":
    main()
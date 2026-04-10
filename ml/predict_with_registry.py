from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REGISTRY_PATH = Path("/opt/project/outputs/model_registry.json")
DEFAULT_INPUT = Path("/opt/project/data/gold/customer_features")
DEFAULT_OUTPUT = Path("/opt/project/outputs/predictions")


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry file not found: {REGISTRY_PATH}")
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_production_entry(model_name: str) -> dict:
    registry = load_registry()

    if "models" not in registry or model_name not in registry["models"]:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    model_info = registry["models"][model_name]
    production_version = model_info.get("production_version")

    if not production_version:
        raise ValueError(f"No production version set for model '{model_name}'.")

    version_entry = model_info.get("versions", {}).get(production_version)
    if not version_entry:
        raise ValueError(
            f"Production version '{production_version}' not found for model '{model_name}'."
        )

    return version_entry


def find_artifact(artifacts: list[str], keyword: str, required: bool = True) -> str | None:
    for artifact in artifacts:
        if keyword.lower() in Path(artifact).name.lower():
            return artifact
    if required:
        raise FileNotFoundError(f"Could not find artifact containing '{keyword}'.")
    return None


def load_customer_features(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        parquet_files = sorted(input_path.glob("*.parquet"))
        if not parquet_files:
            parquet_files = sorted(input_path.rglob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under directory: {input_path}")

        frames = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(frames, ignore_index=True)

    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)

    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)

    raise ValueError(f"Unsupported input path format: {input_path}")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.fillna(0.0)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    required_base_cols = ["txn_count", "total_spend", "avg_amount", "max_amount"]
    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base columns required for feature engineering: {missing}")

    df = df.copy()

    if "spend_per_txn_ratio" not in df.columns:
        df["spend_per_txn_ratio"] = safe_divide(df["total_spend"], df["txn_count"])

    if "max_to_avg_ratio" not in df.columns:
        df["max_to_avg_ratio"] = safe_divide(df["max_amount"], df["avg_amount"])

    if "amount_range" not in df.columns:
        df["amount_range"] = df["max_amount"] - df["avg_amount"]

    if "txn_intensity" not in df.columns:
        df["txn_intensity"] = df["txn_count"] * df["avg_amount"]

    return df


def run_segmentation(input_path: Path, output_dir: Path) -> Path:
    model_name = "customer_segmentation"
    entry = get_production_entry(model_name)

    artifacts = entry["artifacts"]
    kmeans_path = find_artifact(artifacts, "kmeans")
    scaler_path = find_artifact(artifacts, "scaler")

    kmeans = joblib.load(kmeans_path)
    scaler = joblib.load(scaler_path)

    feature_columns = entry["metrics"]["feature_columns"]
    segment_name_map = entry["metrics"].get("segment_name_map", {})

    df = load_customer_features(input_path)
    df = add_engineered_features(df)

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for segmentation: {missing_cols}")

    X = df[feature_columns].copy()
    X_scaled = scaler.transform(X)

    df["predicted_segment"] = kmeans.predict(X_scaled)
    df["predicted_segment_name"] = df["predicted_segment"].astype(str).map(segment_name_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "customer_segmentation_predictions.parquet"
    df.to_parquet(output_path, index=False)

    print("===== Segmentation Prediction Complete =====")
    print(f"Production version used: {entry['version']}")
    print(f"Rows scored: {len(df)}")
    print(f"Saved to: {output_path}")

    return output_path


def run_spend_prediction(input_path: Path, output_dir: Path) -> Path:
    model_name = "spend_prediction"
    entry = get_production_entry(model_name)

    artifacts = entry["artifacts"]
    model_path = find_artifact(artifacts, "random_forest")

    model = joblib.load(model_path)
    metrics = entry["metrics"]

    numeric_cols = metrics["numeric_feature_columns"]
    categorical_cols = metrics["categorical_feature_columns"]
    final_feature_cols = metrics["final_feature_columns"]

    df = load_customer_features(input_path)
    df = add_engineered_features(df)

    # Accept segmentation inference output
    if "customer_segment" not in df.columns and "predicted_segment" in df.columns:
        df["customer_segment"] = df["predicted_segment"]

    if "segment_name" not in df.columns and "predicted_segment_name" in df.columns:
        df["segment_name"] = df["predicted_segment_name"]

    required_cols = numeric_cols + categorical_cols
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for spend prediction: {missing_cols}")

    # Clean numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Clean categorical columns
    for col in categorical_cols:
        df[col] = df[col].fillna(-1).astype(str)

    X = df[required_cols].copy()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    for col in final_feature_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    X_encoded = X_encoded[final_feature_cols]

    predictions = model.predict(X_encoded)

    result = df.copy()
    result["predicted_total_spend"] = predictions

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spend_prediction_outputs.parquet"
    result.to_parquet(output_path, index=False)

    print("===== Spend Prediction Inference Complete =====")
    print(f"Production version used: {entry['version']}")
    print(f"Rows scored: {len(result)}")
    print(f"Saved to: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference using production models from registry.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["customer_segmentation", "spend_prediction"],
        help="Model to run from registry",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input dataset path (directory, parquet, or csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory where predictions will be saved",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if args.model == "customer_segmentation":
        run_segmentation(input_path, output_dir)
    elif args.model == "spend_prediction":
        run_spend_prediction(input_path, output_dir)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()

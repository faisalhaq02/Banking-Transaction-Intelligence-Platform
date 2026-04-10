from __future__ import annotations

import json
import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path("/opt/project")
DATA_DIR = BASE_DIR / "data"
GOLD_DIR = DATA_DIR / "gold"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    OUTPUT_DIR / "customer_risk_scores.parquet",
    OUTPUT_DIR / "customer_anomalies.parquet",
    OUTPUT_DIR / "customer_segments.parquet",
    GOLD_DIR / "customer_features.parquet",
    GOLD_DIR / "customer_features.csv",
]

TARGET_COL = "total_spend"

# Leakage-safe numeric features
NUMERIC_FEATURE_COLS = [
    "txn_count",
    "avg_amount",
    "max_amount",
    "max_to_avg_ratio",
    "amount_range",
]

# Categorical feature(s)
CATEGORICAL_FEATURE_COLS = [
    "customer_segment",
]

MODEL_FILE = MODEL_DIR / "spend_prediction_random_forest.pkl"
METRICS_FILE = OUTPUT_DIR / "spend_prediction_metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "spend_predictions.parquet"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "spend_feature_importance.csv"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "spend_feature_importance.png"


def load_input_data() -> pd.DataFrame:
    for path in INPUT_CANDIDATES:
        if path.exists():
            print(f"Loading input data from: {path}")
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find input data in: {INPUT_CANDIDATES}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    print("===== Spend Prediction Training Started =====")

    df = load_input_data().copy()
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if "customer_id" not in df.columns:
        raise ValueError("customer_id column is required")

    # Keep one row per customer
    df = df.drop_duplicates(subset=["customer_id"]).copy()

    required_cols = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean numeric columns
    for col in NUMERIC_FEATURE_COLS + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Clean categorical columns
    for col in CATEGORICAL_FEATURE_COLS:
        df[col] = df[col].fillna(-1).astype(str)

    # Build feature matrix
    X_num = df[NUMERIC_FEATURE_COLS].copy()
    X_cat = pd.get_dummies(
        df[CATEGORICAL_FEATURE_COLS],
        prefix=CATEGORICAL_FEATURE_COLS,
        drop_first=False,
    )
    X = pd.concat([X_num, X_cat], axis=1)

    y = df[TARGET_COL].copy()

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": rmse(y_test, y_pred),
        "r2": float(r2_score(y_test, y_pred)),
        "target_column": TARGET_COL,
        "numeric_feature_columns": NUMERIC_FEATURE_COLS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLS,
        "final_feature_columns": list(X.columns),
        "model_type": "RandomForestRegressor",
        "note": "Leakage-safe version excluding spend_per_txn_ratio and txn_intensity.",
    }

    pred_df = df.loc[idx_test].copy()
    pred_df["actual_total_spend"] = y_test.values
    pred_df["predicted_total_spend"] = y_pred
    pred_df["prediction_error"] = pred_df["predicted_total_spend"] - pred_df["actual_total_spend"]
    pred_df["absolute_error"] = pred_df["prediction_error"].abs()
    pred_df = pred_df.sort_values("absolute_error", ascending=False)

    importance_df = pd.DataFrame({
        "feature": list(X.columns),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    joblib.dump(model, MODEL_FILE)
    pred_df.to_parquet(PREDICTIONS_FILE, index=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Spend Prediction Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n===== Spend Prediction Complete =====")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved metrics to: {METRICS_FILE}")
    print(f"Saved predictions to: {PREDICTIONS_FILE}")
    print(f"Saved feature importance csv to: {FEATURE_IMPORTANCE_FILE}")
    print(f"Saved feature importance plot to: {FEATURE_IMPORTANCE_PLOT}")

    print("\n===== Metrics =====")
    print(json.dumps(metrics, indent=2))

    print("\n===== Feature Importance =====")
    print(importance_df.to_string(index=False))


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


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

ANOMALY_OUTPUT_FILE = OUTPUT_DIR / "customer_anomalies.parquet"
TOP_ANOMALIES_FILE = OUTPUT_DIR / "top_unusual_customers.csv"
ANOMALY_SUMMARY_FILE = OUTPUT_DIR / "anomaly_summary.json"

IFOREST_MODEL_FILE = MODEL_DIR / "anomaly_isolation_forest.pkl"
LOF_MODEL_FILE = MODEL_DIR / "anomaly_lof.pkl"
OCSVM_MODEL_FILE = MODEL_DIR / "anomaly_ocsvm.pkl"
ANOMALY_SCALER_FILE = MODEL_DIR / "anomaly_scaler.pkl"


def load_input_data() -> pd.DataFrame:
    for path in INPUT_CANDIDATES:
        if path.exists():
            print(f"Loading input data from: {path}")
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find input data in: {INPUT_CANDIDATES}")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    df = df.copy()

    required = ["txn_count", "total_spend", "avg_amount", "max_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    customer_id_col = "customer_id" if "customer_id" in df.columns else None
    if customer_id_col is None:
        df["customer_id"] = np.arange(1, len(df) + 1)
        customer_id_col = "customer_id"

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    if "spend_per_txn_ratio" not in df.columns:
        df["spend_per_txn_ratio"] = np.where(
            df["txn_count"] > 0,
            df["total_spend"] / df["txn_count"],
            0,
        )

    if "max_to_avg_ratio" not in df.columns:
        df["max_to_avg_ratio"] = np.where(
            df["avg_amount"] > 0,
            df["max_amount"] / df["avg_amount"],
            0,
        )

    if "amount_range" not in df.columns:
        df["amount_range"] = df["max_amount"] - df["avg_amount"]

    if "txn_intensity" not in df.columns:
        df["txn_intensity"] = df["txn_count"] * df["avg_amount"]

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df, FEATURE_COLS, customer_id_col


def percentile_anomaly_score(raw_values: np.ndarray, higher_more_anomalous: bool = True) -> np.ndarray:
    s = pd.Series(raw_values)
    if higher_more_anomalous:
        return s.rank(pct=True, method="average").to_numpy()
    return (-s).rank(pct=True, method="average").to_numpy()


def severity_from_votes(votes: int) -> str:
    if votes <= 0:
        return "Normal"
    if votes == 1:
        return "Mildly Unusual"
    if votes == 2:
        return "Strongly Unusual"
    return "Highly Unusual"


def main() -> None:
    print("===== Multi-Model Customer Anomaly Detection Started =====")

    df = load_input_data()
    print(f"Loaded shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    df, feature_cols, customer_id_col = prepare_features(df)
    df = df.drop_duplicates(subset=[customer_id_col]).copy()

    X = df[feature_cols].copy()
    print(f"Prepared feature shape: {X.shape}")
    print(f"Feature columns: {feature_cols}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = 0.03

    # Isolation Forest
    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    iforest.fit(X_scaled)
    iforest_pred = iforest.predict(X_scaled)  # -1 anomaly, 1 normal
    iforest_raw = -iforest.decision_function(X_scaled)
    df["iforest_flag"] = np.where(iforest_pred == -1, 1, 0)
    df["iforest_score"] = percentile_anomaly_score(iforest_raw, higher_more_anomalous=True)

    # LOF
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=False,
    )
    lof_pred = lof.fit_predict(X_scaled)  # -1 anomaly, 1 normal
    lof_raw = -lof.negative_outlier_factor_
    df["lof_flag"] = np.where(lof_pred == -1, 1, 0)
    df["lof_score"] = percentile_anomaly_score(lof_raw, higher_more_anomalous=True)

    # One-Class SVM
    ocsvm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=contamination,
    )
    ocsvm.fit(X_scaled)
    ocsvm_pred = ocsvm.predict(X_scaled)  # -1 anomaly, 1 normal
    ocsvm_raw = -ocsvm.decision_function(X_scaled).ravel()
    df["ocsvm_flag"] = np.where(ocsvm_pred == -1, 1, 0)
    df["ocsvm_score"] = percentile_anomaly_score(ocsvm_raw, higher_more_anomalous=True)

    # Ensemble
    df["ensemble_anomaly_votes"] = (
        df["iforest_flag"] + df["lof_flag"] + df["ocsvm_flag"]
    )
    df["ensemble_anomaly_flag"] = np.where(df["ensemble_anomaly_votes"] >= 2, 1, 0)
    df["anomaly_severity"] = df["ensemble_anomaly_votes"].apply(severity_from_votes)

    df["ensemble_anomaly_score"] = (
        0.5 * df["iforest_score"] +
        0.3 * df["lof_score"] +
        0.2 * df["ocsvm_score"]
    )

    df = df.sort_values(
        ["ensemble_anomaly_votes", "ensemble_anomaly_score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    top_cols = [
        customer_id_col,
        "txn_count",
        "total_spend",
        "avg_amount",
        "max_amount",
        "iforest_flag",
        "lof_flag",
        "ocsvm_flag",
        "ensemble_anomaly_votes",
        "ensemble_anomaly_flag",
        "anomaly_severity",
        "ensemble_anomaly_score",
    ]

    if "customer_segment" in df.columns:
        top_cols.insert(1, "customer_segment")
    if "segment_name" in df.columns:
        top_cols.insert(2, "segment_name")

    top_unusual = df[df["ensemble_anomaly_flag"] == 1][top_cols].copy().head(50)

    summary = {
        "total_customers_scored": int(len(df)),
        "feature_columns": feature_cols,
        "contamination": contamination,
        "iforest_anomalies": int(df["iforest_flag"].sum()),
        "lof_anomalies": int(df["lof_flag"].sum()),
        "ocsvm_anomalies": int(df["ocsvm_flag"].sum()),
        "ensemble_anomalies": int(df["ensemble_anomaly_flag"].sum()),
        "ensemble_anomaly_rate": float(df["ensemble_anomaly_flag"].mean()),
        "severity_distribution": {
            k: int(v) for k, v in df["anomaly_severity"].value_counts().to_dict().items()
        },
        "ensemble_rule": "ensemble_anomaly_flag = 1 when at least 2 of 3 models flag anomaly",
        "ensemble_score_weights": {
            "iforest_score": 0.5,
            "lof_score": 0.3,
            "ocsvm_score": 0.2,
        },
    }

    df.to_parquet(ANOMALY_OUTPUT_FILE, index=False)
    top_unusual.to_csv(TOP_ANOMALIES_FILE, index=False)

    joblib.dump(iforest, IFOREST_MODEL_FILE)
    # LOF is fit-only with novelty=False, but saving the fitted object is still fine for documentation/project reuse
    joblib.dump(lof, LOF_MODEL_FILE)
    joblib.dump(ocsvm, OCSVM_MODEL_FILE)
    joblib.dump(scaler, ANOMALY_SCALER_FILE)

    with open(ANOMALY_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Multi-Model Anomaly Detection Complete =====")
    print(f"Saved anomaly output to: {ANOMALY_OUTPUT_FILE}")
    print(f"Saved top unusual customers to: {TOP_ANOMALIES_FILE}")
    print(f"Saved anomaly summary to: {ANOMALY_SUMMARY_FILE}")
    print(f"Saved Isolation Forest model to: {IFOREST_MODEL_FILE}")
    print(f"Saved LOF model to: {LOF_MODEL_FILE}")
    print(f"Saved One-Class SVM model to: {OCSVM_MODEL_FILE}")
    print(f"Saved scaler to: {ANOMALY_SCALER_FILE}")

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2))

    if not top_unusual.empty:
        print("\nTop unusual customers preview:")
        print(top_unusual.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
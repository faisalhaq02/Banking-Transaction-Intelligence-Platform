from __future__ import annotations

import os
from pathlib import Path

# ==================================================
# Base project paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
BI_EXPORTS_DIR = BASE_DIR / "bi_exports"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
METRICS_DIR = BASE_DIR / "metrics"
LOGS_DIR = BASE_DIR / "logs"

# Local lake folders
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Streaming folders
STREAM_BATCHES_DIR = DATA_DIR / "stream_batches"
STREAM_BATCHES_UPLOADED_DIR = DATA_DIR / "stream_batches_uploaded"

# ==================================================
# General settings
# ==================================================
TOP_N_DEFAULT = 5
MAX_ROWS_PREVIEW = 10

# ==================================================
# Azure / Cloud settings
# ==================================================
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()

AZURE_BI_CONTAINER = os.getenv("AZURE_BI_CONTAINER", "bi-exports").strip()
AZURE_OUTPUTS_CONTAINER = os.getenv("AZURE_OUTPUTS_CONTAINER", "models").strip()
AZURE_ICEBERG_CONTAINER = os.getenv("AZURE_ICEBERG_CONTAINER", "iceberg").strip()
AZURE_BRONZE_CONTAINER = os.getenv("AZURE_BRONZE_CONTAINER", "bronze").strip()
AZURE_SILVER_CONTAINER = os.getenv("AZURE_SILVER_CONTAINER", "silver").strip()
AZURE_GOLD_CONTAINER = os.getenv("AZURE_GOLD_CONTAINER", "gold").strip()

# Optional blob prefixes
AZURE_BI_PREFIX = os.getenv("AZURE_BI_PREFIX", "").strip()
AZURE_OUTPUTS_PREFIX = os.getenv("AZURE_OUTPUTS_PREFIX", "").strip()
AZURE_STREAMING_PREFIX = os.getenv("AZURE_STREAMING_PREFIX", "streaming_transactions").strip()

if AZURE_BI_PREFIX and not AZURE_BI_PREFIX.endswith("/"):
    AZURE_BI_PREFIX += "/"

if AZURE_OUTPUTS_PREFIX and not AZURE_OUTPUTS_PREFIX.endswith("/"):
    AZURE_OUTPUTS_PREFIX += "/"

if AZURE_STREAMING_PREFIX and not AZURE_STREAMING_PREFIX.endswith("/"):
    AZURE_STREAMING_PREFIX += "/"

# ==================================================
# Azure container mapping
# ==================================================
CONTAINERS = {
    "bronze": AZURE_BRONZE_CONTAINER,
    "silver": AZURE_SILVER_CONTAINER,
    "gold": AZURE_GOLD_CONTAINER,
    "models": AZURE_OUTPUTS_CONTAINER,
    "iceberg": AZURE_ICEBERG_CONTAINER,
    "bi_exports": AZURE_BI_CONTAINER,
}

# ==================================================
# Azure blob paths
# Used by cloud_data_access.py
# ==================================================
AZURE_BLOB_PATHS = {
    # ----------------------------
    # BI exports (CSV)
    # ----------------------------
    "executive_kpis_csv": f"{AZURE_BI_PREFIX}executive_kpis.csv",
    "risk_anomaly_summary_csv": f"{AZURE_BI_PREFIX}risk_anomaly_summary.csv",
    "segment_summary_csv": f"{AZURE_BI_PREFIX}segment_summary.csv",
    "spend_prediction_summary_csv": f"{AZURE_BI_PREFIX}spend_prediction_summary.csv",
    "channel_summary_csv": f"{AZURE_BI_PREFIX}channel_summary.csv",
    "merchant_summary_csv": f"{AZURE_BI_PREFIX}merchant_summary.csv",
    "geo_summary_csv": f"{AZURE_BI_PREFIX}geo_summary.csv",
    "transaction_trend_summary_csv": f"{AZURE_BI_PREFIX}transaction_trend_summary.csv",

    # ----------------------------
    # BI exports (Parquet)
    # ----------------------------
    "executive_kpis_parquet": f"{AZURE_BI_PREFIX}executive_kpis.parquet",
    "risk_anomaly_summary_parquet": f"{AZURE_BI_PREFIX}risk_anomaly_summary.parquet",
    "segment_summary_parquet": f"{AZURE_BI_PREFIX}segment_summary.parquet",
    "spend_prediction_summary_parquet": f"{AZURE_BI_PREFIX}spend_prediction_summary.parquet",
    "channel_summary_parquet": f"{AZURE_BI_PREFIX}channel_summary.parquet",
    "merchant_summary_parquet": f"{AZURE_BI_PREFIX}merchant_summary.parquet",
    "geo_summary_parquet": f"{AZURE_BI_PREFIX}geo_summary.parquet",
    "transaction_trend_summary_parquet": f"{AZURE_BI_PREFIX}transaction_trend_summary.parquet",

    # ----------------------------
    # ML / model outputs
    # ----------------------------
    "risk_scores": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores.parquet",
    "anomalies": f"{AZURE_OUTPUTS_PREFIX}customer_anomalies.parquet",
    "segments": f"{AZURE_OUTPUTS_PREFIX}customer_segments.parquet",
    "retraining_decision": f"{AZURE_OUTPUTS_PREFIX}retraining_decision.json",
    "risk_bucket_summary": f"{AZURE_OUTPUTS_PREFIX}risk_bucket_summary.json",
    "anomaly_summary": f"{AZURE_OUTPUTS_PREFIX}anomaly_summary.json",

    # ----------------------------
    # Enriched risk / anomaly outputs
    # ----------------------------
    "customer_risk_scores_enriched": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "customer_risk_scores_enriched_csv": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "customer_risk_scores_enriched_parquet": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores.parquet",

    # Spend prediction aliases
    "spend_prediction_outputs": f"{AZURE_BI_PREFIX}spend_prediction_summary.parquet",
    "spend_prediction_outputs_parquet": f"{AZURE_BI_PREFIX}spend_prediction_summary.parquet",
    "spend_prediction_outputs_csv": f"{AZURE_BI_PREFIX}spend_prediction_summary.csv",

    # Anomaly aliases
    "anomaly_scores": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "anomaly_scores_parquet": f"{AZURE_OUTPUTS_PREFIX}customer_anomalies.parquet",
    "anomaly_scores_csv": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "anomaly_outputs": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "anomaly_output": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "anomaly_predictions": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "anomaly_results": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "fraud_anomalies": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "fraud_predictions": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "fraud_risk_outputs": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "fraud_risk_outputs_parquet": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores.parquet",
    "fraud_risk_outputs_csv": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "customer_anomaly_summary": f"{AZURE_BI_PREFIX}risk_anomaly_summary.parquet",
    "customer_anomaly_summary_parquet": f"{AZURE_BI_PREFIX}risk_anomaly_summary.parquet",
    "customer_anomaly_summary_csv": f"{AZURE_BI_PREFIX}risk_anomaly_summary.csv",
    "transaction_anomalies": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "transaction_anomalies_parquet": f"{AZURE_OUTPUTS_PREFIX}customer_anomalies.parquet",
    "transaction_anomalies_csv": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "suspicious_transactions": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "suspicious_transaction_exports": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "anomaly_transactions_enriched": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",
    "fraud_transactions_enriched": f"{AZURE_OUTPUTS_PREFIX}customer_risk_scores_enriched.csv",

    # ----------------------------
    # Streaming / lake placeholders
    # ----------------------------
    "latest_stream_batch_csv": f"{AZURE_STREAMING_PREFIX}latest_transactions.csv",
    "latest_stream_batch_parquet": f"{AZURE_STREAMING_PREFIX}latest_transactions.parquet",
    "silver_transactions_parquet": f"{AZURE_STREAMING_PREFIX}silver_transactions.parquet",
    "silver_transactions_csv": f"{AZURE_STREAMING_PREFIX}silver_transactions.csv",
    "gold_transactions_parquet": f"{AZURE_STREAMING_PREFIX}gold_transactions.parquet",
    "gold_transactions_csv": f"{AZURE_STREAMING_PREFIX}gold_transactions.csv",
}

# ==================================================
# Local file paths
# ==================================================
DATA_PATHS = {
    # ----------------------------
    # ML / AI outputs
    # ----------------------------
    "risk_scores": OUTPUTS_DIR / "customer_risk_scores.parquet",
    "anomalies": OUTPUTS_DIR / "customer_anomalies.parquet",
    "segments": OUTPUTS_DIR / "customer_segments.parquet",
    "retraining_decision": OUTPUTS_DIR / "retraining_decision.json",
    "risk_bucket_summary": OUTPUTS_DIR / "risk_bucket_summary.json",
    "anomaly_summary": OUTPUTS_DIR / "anomaly_summary.json",

    # ----------------------------
    # Enriched customer risk / anomaly files
    # ----------------------------
    "customer_risk_scores_enriched": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "customer_risk_scores_enriched_csv": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "customer_risk_scores_enriched_parquet": OUTPUTS_DIR / "customer_risk_scores.parquet",

    # Spend prediction aliases
    "spend_prediction_outputs": BI_EXPORTS_DIR / "spend_prediction_summary.parquet",
    "spend_prediction_outputs_parquet": BI_EXPORTS_DIR / "spend_prediction_summary.parquet",
    "spend_prediction_outputs_csv": BI_EXPORTS_DIR / "spend_prediction_summary.csv",

    # Anomaly aliases
    "anomaly_scores": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "anomaly_scores_parquet": OUTPUTS_DIR / "customer_anomalies.parquet",
    "anomaly_scores_csv": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "anomaly_outputs": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "anomaly_output": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "anomaly_predictions": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "anomaly_results": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "fraud_anomalies": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "fraud_predictions": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "fraud_risk_outputs": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "fraud_risk_outputs_parquet": OUTPUTS_DIR / "customer_risk_scores.parquet",
    "fraud_risk_outputs_csv": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "customer_anomaly_summary": BI_EXPORTS_DIR / "risk_anomaly_summary.parquet",
    "customer_anomaly_summary_parquet": BI_EXPORTS_DIR / "risk_anomaly_summary.parquet",
    "customer_anomaly_summary_csv": BI_EXPORTS_DIR / "risk_anomaly_summary.csv",
    "transaction_anomalies": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "transaction_anomalies_parquet": OUTPUTS_DIR / "customer_anomalies.parquet",
    "transaction_anomalies_csv": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "suspicious_transactions": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "suspicious_transaction_exports": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "anomaly_transactions_enriched": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",
    "fraud_transactions_enriched": OUTPUTS_DIR / "customer_risk_scores_enriched.csv",

    # ----------------------------
    # BI exports - CSV
    # ----------------------------
    "executive_kpis_csv": BI_EXPORTS_DIR / "executive_kpis.csv",
    "risk_anomaly_summary_csv": BI_EXPORTS_DIR / "risk_anomaly_summary.csv",
    "segment_summary_csv": BI_EXPORTS_DIR / "segment_summary.csv",
    "spend_prediction_summary_csv": BI_EXPORTS_DIR / "spend_prediction_summary.csv",
    "channel_summary_csv": BI_EXPORTS_DIR / "channel_summary.csv",
    "merchant_summary_csv": BI_EXPORTS_DIR / "merchant_summary.csv",
    "geo_summary_csv": BI_EXPORTS_DIR / "geo_summary.csv",
    "transaction_trend_summary_csv": BI_EXPORTS_DIR / "transaction_trend_summary.csv",

    # ----------------------------
    # BI exports - Parquet
    # ----------------------------
    "executive_kpis_parquet": BI_EXPORTS_DIR / "executive_kpis.parquet",
    "risk_anomaly_summary_parquet": BI_EXPORTS_DIR / "risk_anomaly_summary.parquet",
    "segment_summary_parquet": BI_EXPORTS_DIR / "segment_summary.parquet",
    "spend_prediction_summary_parquet": BI_EXPORTS_DIR / "spend_prediction_summary.parquet",
    "channel_summary_parquet": BI_EXPORTS_DIR / "channel_summary.parquet",
    "merchant_summary_parquet": BI_EXPORTS_DIR / "merchant_summary.parquet",
    "geo_summary_parquet": BI_EXPORTS_DIR / "geo_summary.parquet",
    "transaction_trend_summary_parquet": BI_EXPORTS_DIR / "transaction_trend_summary.parquet",

    # ----------------------------
    # Streaming / batch folders
    # ----------------------------
    "stream_batches_dir": STREAM_BATCHES_DIR,
    "stream_batches_uploaded_dir": STREAM_BATCHES_UPLOADED_DIR,

    # Latest stream files
    "latest_batch_csv": STREAM_BATCHES_DIR / "latest_transactions.csv",
    "latest_batch_parquet": STREAM_BATCHES_DIR / "latest_transactions.parquet",

    # Lakehouse-style folders
    "bronze_dir": BRONZE_DIR,
    "silver_dir": SILVER_DIR,
    "gold_dir": GOLD_DIR,

    # Silver / gold transaction tables
    "silver_transactions_parquet": SILVER_DIR / "silver_transactions.parquet",
    "silver_transactions_csv": SILVER_DIR / "silver_transactions.csv",
    "gold_transactions_parquet": GOLD_DIR / "gold_transactions.parquet",
    "gold_transactions_csv": GOLD_DIR / "gold_transactions.csv",

    # Optional flat fallback files
    "silver_transactions_data_parquet": DATA_DIR / "silver_transactions.parquet",
    "silver_transactions_data_csv": DATA_DIR / "silver_transactions.csv",
    "gold_transactions_data_parquet": DATA_DIR / "gold_transactions.parquet",
    "gold_transactions_data_csv": DATA_DIR / "gold_transactions.csv",

    # Logs / metrics
    "metrics_dir": METRICS_DIR,
    "logs_dir": LOGS_DIR,
}

# ==================================================
# Preferred read order
# ==================================================
PREFERRED_PATHS = {
    "executive_kpis": [
        DATA_PATHS["executive_kpis_parquet"],
        DATA_PATHS["executive_kpis_csv"],
        DATA_PATHS["segment_summary_parquet"],
        DATA_PATHS["segment_summary_csv"],
        DATA_PATHS["channel_summary_parquet"],
        DATA_PATHS["channel_summary_csv"],
        DATA_PATHS["merchant_summary_parquet"],
        DATA_PATHS["merchant_summary_csv"],
        DATA_PATHS["geo_summary_parquet"],
        DATA_PATHS["geo_summary_csv"],
        DATA_PATHS["spend_prediction_summary_parquet"],
        DATA_PATHS["spend_prediction_summary_csv"],
        DATA_PATHS["transaction_trend_summary_parquet"],
        DATA_PATHS["transaction_trend_summary_csv"],
    ],
    "risk_anomaly_summary": [
        DATA_PATHS["risk_anomaly_summary_parquet"],
        DATA_PATHS["risk_anomaly_summary_csv"],
        DATA_PATHS["anomalies"],
        DATA_PATHS["risk_scores"],
    ],
    "anomaly_outputs": [
        DATA_PATHS["transaction_anomalies"],
        DATA_PATHS["transaction_anomalies_parquet"],
        DATA_PATHS["transaction_anomalies_csv"],
        DATA_PATHS["anomaly_outputs"],
        DATA_PATHS["anomaly_scores"],
        DATA_PATHS["anomaly_scores_csv"],
        DATA_PATHS["fraud_risk_outputs"],
        DATA_PATHS["fraud_risk_outputs_csv"],
        DATA_PATHS["customer_risk_scores_enriched"],
        DATA_PATHS["customer_anomaly_summary_parquet"],
        DATA_PATHS["customer_anomaly_summary_csv"],
        DATA_PATHS["transaction_trend_summary_parquet"],
        DATA_PATHS["transaction_trend_summary_csv"],
        DATA_PATHS["silver_transactions_parquet"],
        DATA_PATHS["silver_transactions_csv"],
        DATA_PATHS["silver_transactions_data_parquet"],
        DATA_PATHS["silver_transactions_data_csv"],
        DATA_PATHS["latest_batch_parquet"],
        DATA_PATHS["latest_batch_csv"],
    ],
    "segment_summary": [
        DATA_PATHS["segment_summary_parquet"],
        DATA_PATHS["segment_summary_csv"],
        DATA_PATHS["segments"],
    ],
    "spend_prediction_summary": [
        DATA_PATHS["spend_prediction_summary_parquet"],
        DATA_PATHS["spend_prediction_summary_csv"],
        DATA_PATHS["spend_prediction_outputs"],
        DATA_PATHS["spend_prediction_outputs_csv"],
    ],
    "channel_summary": [
        DATA_PATHS["channel_summary_parquet"],
        DATA_PATHS["channel_summary_csv"],
    ],
    "merchant_summary": [
        DATA_PATHS["merchant_summary_parquet"],
        DATA_PATHS["merchant_summary_csv"],
    ],
    "geo_summary": [
        DATA_PATHS["geo_summary_parquet"],
        DATA_PATHS["geo_summary_csv"],
    ],
    "streaming_transactions": [
        DATA_PATHS["silver_transactions_parquet"],
        DATA_PATHS["silver_transactions_csv"],
        DATA_PATHS["silver_transactions_data_parquet"],
        DATA_PATHS["silver_transactions_data_csv"],
        DATA_PATHS["latest_batch_parquet"],
        DATA_PATHS["latest_batch_csv"],
    ],
    "risk_scores": [
        DATA_PATHS["customer_risk_scores_enriched"],
        DATA_PATHS["customer_risk_scores_enriched_csv"],
        DATA_PATHS["risk_scores"],
    ],
}

# ==================================================
# Ensure required folders exist
# ==================================================
for folder in [
    DATA_DIR,
    OUTPUTS_DIR,
    BI_EXPORTS_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    LOGS_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    STREAM_BATCHES_DIR,
    STREAM_BATCHES_UPLOADED_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
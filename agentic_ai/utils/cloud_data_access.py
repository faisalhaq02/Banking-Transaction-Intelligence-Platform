from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd

from agentic_ai.config import (
    AZURE_BI_CONTAINER,
    AZURE_BLOB_PATHS,
    AZURE_OUTPUTS_CONTAINER,
    AZURE_STORAGE_CONNECTION_STRING,
    DATA_PATHS,
)

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    BlobServiceClient = None


def _blob_service_client():
    if not AZURE_STORAGE_CONNECTION_STRING or BlobServiceClient is None:
        return None
    try:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    except Exception:
        return None


def _download_blob_bytes(container_name: str, blob_name: str) -> Optional[bytes]:
    client = _blob_service_client()
    if client is None:
        return None

    try:
        blob_client = client.get_blob_client(container=container_name, blob=blob_name)
        return blob_client.download_blob().readall()
    except Exception:
        return None


def _load_local_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _load_local_parquet(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def _load_blob_csv(container_name: str, blob_name: str) -> pd.DataFrame | None:
    try:
        content = _download_blob_bytes(container_name, blob_name)
        if content is None:
            return None
        return pd.read_csv(BytesIO(content))
    except Exception:
        return None


def _load_blob_parquet(container_name: str, blob_name: str) -> pd.DataFrame | None:
    try:
        content = _download_blob_bytes(container_name, blob_name)
        if content is None:
            return None
        return pd.read_parquet(BytesIO(content))
    except Exception:
        return None


def _cloud_first_csv(local_path_key: str, blob_path_key: str, container_name: str) -> pd.DataFrame | None:
    blob_name = AZURE_BLOB_PATHS.get(blob_path_key)
    if blob_name:
        df = _load_blob_csv(container_name, blob_name)
        if df is not None and not df.empty:
            return df

    local_path = DATA_PATHS.get(local_path_key)
    if local_path:
        return _load_local_csv(local_path)

    return None


def _cloud_first_parquet(local_path_key: str, blob_path_key: str, container_name: str) -> pd.DataFrame | None:
    blob_name = AZURE_BLOB_PATHS.get(blob_path_key)
    if blob_name:
        df = _load_blob_parquet(container_name, blob_name)
        if df is not None and not df.empty:
            return df

    local_path = DATA_PATHS.get(local_path_key)
    if local_path:
        return _load_local_parquet(local_path)

    return None


def load_kpis_cloud_first() -> pd.DataFrame | None:
    return _cloud_first_csv("executive_kpis_csv", "executive_kpis_csv", AZURE_BI_CONTAINER)


def load_spend_prediction_cloud_first() -> pd.DataFrame | None:
    return _cloud_first_csv("spend_prediction_summary_csv", "spend_prediction_summary_csv", AZURE_BI_CONTAINER)


def load_spend_predictions_cloud_first() -> pd.DataFrame | None:
    return load_spend_prediction_cloud_first()


def load_channel_summary_cloud_first() -> pd.DataFrame | None:
    return _cloud_first_csv("channel_summary_csv", "channel_summary_csv", AZURE_BI_CONTAINER)


def load_risk_anomaly_summary_cloud_first() -> pd.DataFrame | None:
    return _cloud_first_csv("risk_anomaly_summary_csv", "risk_anomaly_summary_csv", AZURE_BI_CONTAINER)


def load_segment_summary_cloud_first() -> pd.DataFrame | None:
    return _cloud_first_csv("segment_summary_csv", "segment_summary_csv", AZURE_BI_CONTAINER)


def load_risk_cloud_first() -> pd.DataFrame | None:
    return _cloud_first_parquet("risk_scores", "risk_scores", AZURE_OUTPUTS_CONTAINER)


def load_anomalies_cloud_first() -> pd.DataFrame | None:
    return _cloud_first_parquet("anomalies", "anomalies", AZURE_OUTPUTS_CONTAINER)


def load_segments_cloud_first() -> pd.DataFrame | None:
    return _cloud_first_parquet("segments", "segments", AZURE_OUTPUTS_CONTAINER)
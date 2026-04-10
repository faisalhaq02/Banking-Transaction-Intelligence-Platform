from __future__ import annotations

import io
import os
from urllib.parse import quote_plus

import pandas as pd
from azure.storage.blob import BlobServiceClient
from sqlalchemy import create_engine


FILES_TO_LOAD = {
    "bi_exports/channel_summary.csv": "channel_summary",
    "bi_exports/executive_kpis.csv": "executive_kpis",
    "bi_exports/geography_summary.csv": "geography_summary",
    "bi_exports/hourly_activity_summary.csv": "hourly_activity_summary",
    "bi_exports/location_mismatch_summary.csv": "location_mismatch_summary",
    "bi_exports/merchant_category_summary.csv": "merchant_category_summary",
    "bi_exports/risk_anomaly_summary.csv": "risk_anomaly_summary",
    "bi_exports/spend_prediction_summary.csv": "spend_prediction_summary",
    "bi_exports/transaction_trend_summary.csv": "transaction_trend_summary",
}


def get_env_vars() -> dict:
    return {
        "AZURE_STORAGE_CONNECTION_STRING": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        "BI_CONTAINER_NAME": os.getenv("BI_CONTAINER_NAME", "bi-exports"),
        "AZURE_SQL_SERVER": os.getenv("AZURE_SQL_SERVER"),
        "AZURE_SQL_DATABASE": os.getenv("AZURE_SQL_DATABASE"),
        "AZURE_SQL_USERNAME": os.getenv("AZURE_SQL_USERNAME"),
        "AZURE_SQL_PASSWORD": os.getenv("AZURE_SQL_PASSWORD"),
        "AZURE_SQL_ODBC_DRIVER": os.getenv("AZURE_SQL_ODBC_DRIVER", "ODBC Driver 18 for SQL Server"),
    }


def validate_env(env: dict) -> None:
    required = {
        "AZURE_STORAGE_CONNECTION_STRING": env["AZURE_STORAGE_CONNECTION_STRING"],
        "AZURE_SQL_SERVER": env["AZURE_SQL_SERVER"],
        "AZURE_SQL_DATABASE": env["AZURE_SQL_DATABASE"],
        "AZURE_SQL_USERNAME": env["AZURE_SQL_USERNAME"],
        "AZURE_SQL_PASSWORD": env["AZURE_SQL_PASSWORD"],
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")


def get_sql_engine(env: dict):
    odbc_str = (
        f"DRIVER={{{env['AZURE_SQL_ODBC_DRIVER']}}};"
        f"SERVER={env['AZURE_SQL_SERVER']},1433;"
        f"DATABASE={env['AZURE_SQL_DATABASE']};"
        f"UID={env['AZURE_SQL_USERNAME']};"
        f"PWD={env['AZURE_SQL_PASSWORD']};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    connection_url = "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_str)

    return create_engine(
        connection_url,
        fast_executemany=True,
        pool_pre_ping=True,
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    return df


def download_blob_to_dataframe(
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_name: str,
) -> pd.DataFrame:
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )
    blob_data = blob_client.download_blob().readall()

    if blob_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(blob_data))

    if blob_name.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(blob_data))

    raise ValueError(f"Unsupported file type: {blob_name}")


def load_dataframe_to_sql(df: pd.DataFrame, table_name: str, engine) -> None:
    df = normalize_columns(df)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=200,
        method=None,
    )


def main() -> None:
    env = get_env_vars()
    validate_env(env)

    print("Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(
        env["AZURE_STORAGE_CONNECTION_STRING"]
    )

    print("Connecting to Azure SQL...")
    engine = get_sql_engine(env)

    for blob_name, table_name in FILES_TO_LOAD.items():
        try:
            print(f"Reading blob: {blob_name}")
            df = download_blob_to_dataframe(
                blob_service_client,
                env["BI_CONTAINER_NAME"],
                blob_name,
            )

            if df.empty:
                print(f"Skipped {blob_name}: empty file")
                continue

            print(f"Loading {blob_name} -> {table_name} ({len(df)} rows)")
            load_dataframe_to_sql(df, table_name, engine)
            print(f"Done: {table_name}")

        except Exception as exc:
            print(f"Failed for {blob_name}: {exc}")
            raise


if __name__ == "__main__":
    main()
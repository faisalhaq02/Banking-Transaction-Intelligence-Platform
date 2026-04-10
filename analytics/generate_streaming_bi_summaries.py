from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


STREAM_BATCH_DIR = Path("/opt/project/data/stream_batches")
EXPORT_DIR = Path("/opt/project/bi_exports")


def load_recent_stream_batches(max_files: int = 20) -> pd.DataFrame:
    if not STREAM_BATCH_DIR.exists():
        raise FileNotFoundError(f"Stream batch directory not found: {STREAM_BATCH_DIR}")

    batch_files = sorted(
        STREAM_BATCH_DIR.glob("transactions_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not batch_files:
        raise FileNotFoundError(f"No stream batch files found in {STREAM_BATCH_DIR}")

    selected_files = batch_files[:max_files]
    print(f"[OK] Reading {len(selected_files)} recent stream batch files")

    records = []
    for file_path in selected_files:
        print(f"[OK] Reading stream batch: {file_path}")
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

    if not records:
        raise ValueError("No records found in selected stream batch files")

    df = pd.DataFrame(records)
    print(f"[OK] Loaded {len(df)} streamed transactions from {len(selected_files)} files")
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["transaction_date"] = df["timestamp"].dt.date.astype("string")
        df["transaction_hour"] = df["timestamp"].dt.hour

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    if "city" in df.columns and "home_city" in df.columns:
        df["city_mismatch"] = (df["city"].fillna("").astype(str) != df["home_city"].fillna("").astype(str)).astype(int)

    if "country" in df.columns and "home_country" in df.columns:
        df["country_mismatch"] = (
            df["country"].fillna("").astype(str) != df["home_country"].fillna("").astype(str)
        ).astype(int)

    if "city_mismatch" in df.columns or "country_mismatch" in df.columns:
        df["location_mismatch_flag"] = (
            df.get("city_mismatch", 0).fillna(0).astype(int) |
            df.get("country_mismatch", 0).fillna(0).astype(int)
        ).astype(int)

    return df


def export_transaction_trend_summary(df: pd.DataFrame) -> None:
    needed = {"transaction_date", "amount"}
    if not needed.issubset(df.columns):
        print("[WARN] Skipping transaction_trend_summary: missing required columns")
        return

    out = (
        df.groupby(["transaction_date"], dropna=False)
        .agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )

    out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce")
    out = out.sort_values("transaction_date")

    out.to_csv(EXPORT_DIR / "transaction_trend_summary.csv", index=False)
    out.to_parquet(EXPORT_DIR / "transaction_trend_summary.parquet", index=False)
    print("[OK] transaction_trend_summary exported")


def export_channel_summary(df: pd.DataFrame) -> None:
    if "channel" not in df.columns or "amount" not in df.columns:
        print("[WARN] Skipping channel_summary: missing required columns")
        return

    out = (
        df.groupby(["channel"], dropna=False)
        .agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )

    out.to_csv(EXPORT_DIR / "channel_summary.csv", index=False)
    out.to_parquet(EXPORT_DIR / "channel_summary.parquet", index=False)
    print("[OK] channel_summary exported")


def export_geography_summary(df: pd.DataFrame) -> None:
    needed = {"country", "city", "amount"}
    if not needed.issubset(df.columns):
        print("[WARN] Skipping geography_summary: missing required columns")
        return

    group_cols = ["country", "city"]
    if "location_mismatch_flag" in df.columns:
        group_cols.append("location_mismatch_flag")

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )

    out.to_csv(EXPORT_DIR / "geography_summary.csv", index=False)
    out.to_parquet(EXPORT_DIR / "geography_summary.parquet", index=False)
    print("[OK] geography_summary exported")


def export_merchant_category_summary(df: pd.DataFrame) -> None:
    needed = {"merchant_category", "amount"}
    if not needed.issubset(df.columns):
        print("[WARN] Skipping merchant_category_summary: missing required columns")
        return

    group_cols = ["merchant_category"]
    if "merchant_risk_level" in df.columns:
        group_cols.append("merchant_risk_level")

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )

    out.to_csv(EXPORT_DIR / "merchant_category_summary.csv", index=False)
    out.to_parquet(EXPORT_DIR / "merchant_category_summary.parquet", index=False)
    print("[OK] merchant_category_summary exported")


def export_location_mismatch_summary(df: pd.DataFrame) -> None:
    if "location_mismatch_flag" not in df.columns or "amount" not in df.columns:
        print("[WARN] Skipping location_mismatch_summary: missing required columns")
        return

    group_cols = ["location_mismatch_flag"]
    for col in ["country", "home_country", "city", "home_city"]:
        if col in df.columns:
            group_cols.append(col)

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )

    out.to_csv(EXPORT_DIR / "location_mismatch_summary.csv", index=False)
    out.to_parquet(EXPORT_DIR / "location_mismatch_summary.parquet", index=False)
    print("[OK] location_mismatch_summary exported")


def export_hourly_activity_summary(df: pd.DataFrame) -> None:
    if "transaction_hour" not in df.columns or "amount" not in df.columns:
        print("[WARN] Skipping hourly_activity_summary: missing required columns")
        return

    out = (
        df.groupby(["transaction_hour"], dropna=False)
        .agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
        .sort_values("transaction_hour")
    )

    out.to_csv(EXPORT_DIR / "hourly_activity_summary.csv", index=False)
    out.to_parquet(EXPORT_DIR / "hourly_activity_summary.parquet", index=False)
    print("[OK] hourly_activity_summary exported")


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_recent_stream_batches(max_files=20)
    df = prepare_dataframe(df)

    export_transaction_trend_summary(df)
    export_channel_summary(df)
    export_geography_summary(df)
    export_merchant_category_summary(df)
    export_location_mismatch_summary(df)
    export_hourly_activity_summary(df)

    print("\n[OK] Streaming BI summaries generation complete.")
    print("[OK] Exported files now in /opt/project/bi_exports:")
    for f in sorted(os.listdir(EXPORT_DIR)):
        print(" -", f)


if __name__ == "__main__":
    main()
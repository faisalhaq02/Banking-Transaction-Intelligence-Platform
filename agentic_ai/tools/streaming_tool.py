from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd

from agentic_ai.utils.data_access import (
    safe_load_parquet,
    safe_load_csv,
    standardize,
    first_matching_column,
)
from agentic_ai.utils.formatter import format_number
from agentic_ai.utils.presentation_formatter import (
    make_section_title,
    make_kv_line,
    make_bullet_list,
    make_empty_message,
)


# =========================================================
# DATA LOADING
# =========================================================
def _get_streaming_paths() -> list[Path]:
    """
    Import config lazily to avoid circular imports.
    Build a deduplicated list of candidate streaming paths.
    """
    from agentic_ai import config

    candidate_paths = []

    if hasattr(config, "PREFERRED_PATHS") and "streaming_transactions" in config.PREFERRED_PATHS:
        preferred = config.PREFERRED_PATHS["streaming_transactions"]
        if isinstance(preferred, (list, tuple)):
            candidate_paths.extend(preferred)
        elif preferred is not None:
            candidate_paths.append(preferred)

    data_path_keys = [
        "silver_transactions_parquet",
        "silver_transactions_csv",
        "silver_transactions_data_parquet",
        "silver_transactions_data_csv",
        "latest_batch_parquet",
        "latest_batch_csv",
        "latest_batch_json",
        "latest_batch_jsonl",
        "streaming_transactions_json",
        "streaming_transactions_jsonl",
    ]

    if hasattr(config, "DATA_PATHS"):
        for key in data_path_keys:
            value = config.DATA_PATHS.get(key)
            if value is not None:
                candidate_paths.append(value)

    common_dirs = [
        Path("data/stream_batches"),
        Path("data/streaming"),
        Path("outputs/streaming"),
        Path("outputs/latest"),
        Path("data/silver"),
    ]

    for directory in common_dirs:
        if directory.exists() and directory.is_dir():
            for pattern in ["*.parquet", "*.csv", "*.json", "*.jsonl"]:
                candidate_paths.extend(sorted(directory.glob(pattern)))

    normalized: list[Path] = []
    seen = set()

    for path in candidate_paths:
        try:
            p = Path(path)
        except Exception:
            continue

        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            normalized.append(p)

    return normalized


def _load_json_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".jsonl":
            df = pd.read_json(path, lines=True)
        else:
            df = pd.read_json(path)

        if df is not None and not df.empty:
            return standardize(df)
    except Exception:
        return None

    return None


def _load_file_by_suffix(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return safe_load_parquet(path)
    if suffix == ".csv":
        return safe_load_csv(path)
    if suffix in {".json", ".jsonl"}:
        return _load_json_file(path)

    return None


def _path_priority_score(path: Path) -> int:
    name = str(path).lower()
    score = 0

    if "stream_batches" in name:
        score += 100
    if "latest_batch" in name:
        score += 90
    if "streaming" in name:
        score += 50
    if name.endswith(".jsonl"):
        score += 20
    if name.endswith(".json"):
        score += 10

    return score


def _load_best_streaming_dataset():
    paths = _get_streaming_paths()
    existing_paths = [p for p in paths if isinstance(p, Path) and p.exists() and p.is_file()]

    if not existing_paths:
        return None, None

    def sort_key(p: Path):
        try:
            return (_path_priority_score(p), p.stat().st_mtime)
        except Exception:
            return (_path_priority_score(p), 0)

    existing_paths = sorted(existing_paths, key=sort_key, reverse=True)

    for path in existing_paths:
        try:
            df = _load_file_by_suffix(path)
        except Exception:
            df = None

        if df is not None and not df.empty:
            return standardize(df), path

    return None, None


# =========================================================
# HELPERS
# =========================================================
def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=False)
        return parsed.dropna()
    except Exception:
        return pd.Series(dtype="datetime64[ns]")


def _safe_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series(dtype="float64")


def _clean_value(value, default: str = "—") -> str:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "nan", "null"}:
        return default
    return text


def _format_amount(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return format_number(float(value))
    except Exception:
        return "—"


def _top_counts(df: pd.DataFrame, col: Optional[str], n: int = 5):
    if not col or col not in df.columns:
        return []
    try:
        counts = df[col].fillna("Unknown").astype(str).value_counts().head(n)
        cleaned = []
        for value, count in counts.items():
            display_value = "—" if str(value).strip().lower() in {"unknown", "none", "nan", "null", ""} else str(value)
            cleaned.append((display_value, count))
        return cleaned
    except Exception:
        return []


def _format_top_counts(title: str, items):
    if not items:
        return None
    bullet_items = [f"{value}: {format_number(count)}" for value, count in items]
    return make_bullet_list(title, bullet_items)


def _extract_limit_from_query(q: str, default: int = 5, max_limit: int = 25) -> int:
    match = re.search(r"\b(last|latest|recent|top)\s+(\d+)\b", q)
    if match:
        try:
            value = int(match.group(2))
            return max(1, min(value, max_limit))
        except Exception:
            return default
    return default


def _extract_amount_threshold(q: str, default: float = 10000.0) -> float:
    patterns = [
        r"above\s+([\d,]+(?:\.\d+)?)",
        r"over\s+([\d,]+(?:\.\d+)?)",
        r">=\s*([\d,]+(?:\.\d+)?)",
        r"greater than\s+([\d,]+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except Exception:
                pass

    return default


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _add_if_present(parts: list[str], label: str, value) -> None:
    cleaned = _clean_value(value)
    if cleaned != "—":
        parts.append(f"{label}: {cleaned}")


def _build_transaction_block(
    row,
    txn_col,
    customer_col,
    amount_col,
    channel_col,
    status_col,
    timestamp_col,
    merchant_category_col=None,
    country_col=None,
    extra_parts: list[str] | None = None,
) -> str:
    title = _clean_value(row.get(customer_col)) if customer_col else "—"
    txn_value = _clean_value(row.get(txn_col)) if txn_col else "—"

    parts = []
    if txn_value != "—" and txn_value != title:
        parts.append(f"Txn: {txn_value}")

    if amount_col:
        amount_text = _format_amount(row.get(amount_col))
        if amount_text != "—":
            parts.append(f"Amount: {amount_text}")

    if channel_col:
        _add_if_present(parts, "Channel", row.get(channel_col))
    if status_col:
        _add_if_present(parts, "Status", row.get(status_col))
    if merchant_category_col:
        _add_if_present(parts, "Merchant Category", row.get(merchant_category_col))
    if country_col:
        _add_if_present(parts, "Country", row.get(country_col))
    if timestamp_col:
        _add_if_present(parts, "Timestamp", row.get(timestamp_col))

    if extra_parts:
        parts.extend([p for p in extra_parts if p])

    return f"- **{title}**\n  " + " | ".join(parts)


def _generate_streaming_key_insight(df: pd.DataFrame, ctx: dict) -> str:
    if df is None or df.empty:
        return "No streaming insight is available."

    amount_col = ctx.get("amount_col")
    channel_col = ctx.get("channel_col")
    status_col = ctx.get("status_col")

    if channel_col and channel_col in df.columns:
        top_channel = df[channel_col].fillna("Unknown").astype(str).value_counts().head(1)
        if not top_channel.empty:
            return f"The latest batch is dominated by **{top_channel.index[0]}** transactions."

    if status_col and status_col in df.columns:
        top_status = df[status_col].fillna("Unknown").astype(str).value_counts().head(1)
        if not top_status.empty:
            return f"Most transactions in the latest batch are currently **{top_status.index[0]}**."

    if amount_col and "_amount_numeric" in df.columns:
        amt = df["_amount_numeric"].dropna()
        if not amt.empty:
            return f"The latest batch carries an average transaction amount of **{format_number(amt.mean())}**."

    return "Streaming data is available, but richer fields would improve the live summary."


# =========================================================
# CONTEXT
# =========================================================
def _prepare_streaming_context():
    df, source_path = _load_best_streaming_dataset()

    if df is None:
        return None

    timestamp_col = first_matching_column(
        df,
        ["timestamp", "event_time", "txn_timestamp", "transaction_time", "datetime", "created_at"],
    )
    txn_col = first_matching_column(df, ["transaction_id", "txn_id", "id", "transactionid"])
    customer_col = first_matching_column(df, ["customer_id", "cust_id", "customer"])
    amount_col = first_matching_column(df, ["amount", "txn_amount", "transaction_amount", "total_amount", "spend"])
    channel_col = first_matching_column(df, ["channel", "txn_channel"])
    status_col = first_matching_column(df, ["status", "txn_status", "transaction_status"])
    merchant_category_col = first_matching_column(df, ["merchant_category", "category", "merchant_type"])
    country_col = first_matching_column(df, ["country", "txn_country"])
    home_country_col = first_matching_column(df, ["home_country", "customer_home_country"])
    city_col = first_matching_column(df, ["city", "txn_city"])
    home_city_col = first_matching_column(df, ["home_city", "customer_home_city"])
    device_id_col = first_matching_column(df, ["device_id", "device"])
    merchant_risk_col = first_matching_column(df, ["merchant_risk_level", "risk_level", "merchant_risk"])
    customer_segment_col = first_matching_column(df, ["customer_segment", "segment", "segment_name"])

    working_df = df.copy()

    if amount_col and amount_col in working_df.columns:
        working_df["_amount_numeric"] = _safe_numeric(working_df[amount_col])
    else:
        working_df["_amount_numeric"] = pd.Series([pd.NA] * len(working_df), index=working_df.index)

    if timestamp_col and timestamp_col in working_df.columns:
        try:
            working_df["_parsed_timestamp"] = pd.to_datetime(working_df[timestamp_col], errors="coerce")
        except Exception:
            working_df["_parsed_timestamp"] = pd.NaT
    else:
        working_df["_parsed_timestamp"] = pd.NaT

    if customer_col and customer_col in working_df.columns:
        try:
            working_df["_customer_txn_count"] = working_df.groupby(customer_col)[customer_col].transform("count")
        except Exception:
            working_df["_customer_txn_count"] = pd.NA
    else:
        working_df["_customer_txn_count"] = pd.NA

    if customer_col and customer_col in working_df.columns and "_amount_numeric" in working_df.columns:
        try:
            working_df["_customer_avg_amount"] = (
                working_df.groupby(customer_col)["_amount_numeric"].transform("mean")
            )
        except Exception:
            working_df["_customer_avg_amount"] = pd.NA
    else:
        working_df["_customer_avg_amount"] = pd.NA

    return {
        "df": working_df,
        "source_path": source_path,
        "timestamp_col": timestamp_col,
        "txn_col": txn_col,
        "customer_col": customer_col,
        "amount_col": amount_col,
        "channel_col": channel_col,
        "status_col": status_col,
        "merchant_category_col": merchant_category_col,
        "country_col": country_col,
        "home_country_col": home_country_col,
        "city_col": city_col,
        "home_city_col": home_city_col,
        "device_id_col": device_id_col,
        "merchant_risk_col": merchant_risk_col,
        "customer_segment_col": customer_segment_col,
    }


# =========================================================
# STREAM SUMMARY
# =========================================================
def get_streaming_summary() -> str:
    ctx = _prepare_streaming_context()

    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    df = ctx["df"]
    source_path = ctx["source_path"]
    timestamp_col = ctx["timestamp_col"]
    customer_col = ctx["customer_col"]
    amount_col = ctx["amount_col"]
    channel_col = ctx["channel_col"]
    status_col = ctx["status_col"]
    merchant_category_col = ctx["merchant_category_col"]
    country_col = ctx["country_col"]
    customer_segment_col = ctx["customer_segment_col"]

    lines = [
        make_section_title("Streaming Overview"),
        make_section_title("Latest Batch Summary"),
    ]

    if source_path is not None:
        lines.append(make_kv_line("Source file", source_path.name))

    lines.append(make_kv_line("Records in latest batch", format_number(len(df))))

    if customer_col and customer_col in df.columns:
        try:
            lines.append(
                make_kv_line(
                    "Unique customers",
                    format_number(df[customer_col].nunique(dropna=True)),
                )
            )
        except Exception:
            pass

    if amount_col and amount_col in df.columns:
        amt = df["_amount_numeric"].dropna()
        if not amt.empty:
            lines.append(make_kv_line("Total streamed amount", format_number(amt.sum())))
            lines.append(make_kv_line("Average transaction amount", format_number(amt.mean())))
            lines.append(make_kv_line("Highest transaction amount", format_number(amt.max())))

    if timestamp_col and timestamp_col in df.columns:
        ts = _parse_timestamp_series(df[timestamp_col])
        if not ts.empty:
            lines.append(make_bullet_list(
                "Time Window",
                [
                    f"Earliest timestamp: {ts.min()}",
                    f"Latest timestamp: {ts.max()}",
                ]
            ))

    for section in [
        _format_top_counts("Top Statuses", _top_counts(df, status_col, n=3)),
        _format_top_counts("Top Channels", _top_counts(df, channel_col, n=3)),
        _format_top_counts("Top Merchant Categories", _top_counts(df, merchant_category_col, n=3)),
        _format_top_counts("Top Countries", _top_counts(df, country_col, n=3)),
        _format_top_counts("Top Customer Segments", _top_counts(df, customer_segment_col, n=3)),
    ]:
        if section:
            lines.append(section)

    lines.append(make_section_title("Key Insight"))
    lines.append(_generate_streaming_key_insight(df, ctx))

    return "\n".join(lines)


def get_latest_timestamp() -> str:
    ctx = _prepare_streaming_context()

    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    df = ctx["df"]
    source_path = ctx["source_path"]
    timestamp_col = ctx["timestamp_col"]

    if timestamp_col is None or timestamp_col not in df.columns:
        return make_empty_message(
            f"No timestamp column was found in the streaming data. Available columns: {list(df.columns)}"
        )

    ts = _parse_timestamp_series(df[timestamp_col])
    if ts.empty:
        return make_empty_message(
            "A timestamp column exists in the streaming data, but no valid timestamp values were found."
        )

    latest_ts = ts.max()

    lines = [make_section_title("Latest Streaming Timestamp")]
    lines.append(make_kv_line("Latest timestamp", str(latest_ts)))
    if source_path is not None:
        lines.append(make_kv_line("Source file", source_path.name))
    return "\n".join(lines)


# =========================================================
# RECENT TRANSACTION VIEWS
# =========================================================
def get_latest_transactions(limit: int = 5, channel_filter: str | None = None) -> str:
    ctx = _prepare_streaming_context()

    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    df = ctx["df"].copy()

    if channel_filter and ctx["channel_col"] and ctx["channel_col"] in df.columns:
        df = df[df[ctx["channel_col"]].fillna("").astype(str).str.upper() == channel_filter.upper()]

    if df.empty:
        if channel_filter:
            return make_empty_message(f"No recent {channel_filter.upper()} transactions were found.")
        return make_empty_message("No recent transactions were found.")

    if "_parsed_timestamp" in df.columns:
        try:
            df = df.sort_values("_parsed_timestamp", ascending=False, na_position="last")
        except Exception:
            pass

    rows = df.head(limit)

    title = f"Latest {limit} Transactions"
    if channel_filter:
        title = f"Latest {limit} {channel_filter.upper()} Transactions"

    lines = [make_section_title(title)]
    for _, row in rows.iterrows():
        lines.append(
            _build_transaction_block(
                row=row,
                txn_col=ctx["txn_col"],
                customer_col=ctx["customer_col"],
                amount_col=ctx["amount_col"],
                channel_col=ctx["channel_col"],
                status_col=ctx["status_col"],
                timestamp_col=ctx["timestamp_col"],
                merchant_category_col=ctx["merchant_category_col"],
                country_col=ctx["country_col"],
            )
        )

    return "\n".join(lines)


# =========================================================
# REAL-TIME SUSPICION SCORING
# =========================================================
def _score_streaming_suspicion(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    working = df.copy()

    amount_col = ctx["amount_col"]
    status_col = ctx["status_col"]
    channel_col = ctx["channel_col"]
    country_col = ctx["country_col"]
    home_country_col = ctx["home_country_col"]
    city_col = ctx["city_col"]
    home_city_col = ctx["home_city_col"]
    merchant_risk_col = ctx["merchant_risk_col"]
    merchant_category_col = ctx["merchant_category_col"]
    customer_col = ctx["customer_col"]

    working["_suspicion_score"] = 0
    working["_suspicion_reasons"] = ""

    def add_reason(mask, points, reason_text):
        if mask is None:
            return
        try:
            m = mask.fillna(False)
        except Exception:
            return

        working.loc[m, "_suspicion_score"] += points
        working.loc[m, "_suspicion_reasons"] = (
            working.loc[m, "_suspicion_reasons"]
            .astype(str)
            .apply(lambda x: f"{x}; {reason_text}".strip("; ").strip())
        )

    amt = working["_amount_numeric"]

    if amount_col and "_amount_numeric" in working.columns:
        add_reason(amt >= 10000, 4, "very high transaction amount >= 10,000")
        add_reason((amt >= 5000) & (amt < 10000), 2, "high transaction amount between 5,000 and 9,999.99")

        try:
            batch_mean = amt.dropna().mean()
            batch_std = amt.dropna().std()
            if pd.notna(batch_mean) and pd.notna(batch_std) and batch_std > 0:
                add_reason(amt >= (batch_mean + 2 * batch_std), 2, "amount is well above batch normal range")
        except Exception:
            pass

    if status_col and status_col in working.columns:
        status_lower = working[status_col].fillna("").astype(str).str.lower()
        add_reason(status_lower.eq("declined"), 2, "transaction was declined")
        add_reason(status_lower.eq("pending"), 1, "transaction is pending")

    if channel_col and channel_col in working.columns:
        channel_lower = working[channel_col].fillna("").astype(str).str.lower()
        add_reason(channel_lower.eq("ecom"), 1, "ecommerce channel")
        add_reason(channel_lower.eq("mobile"), 1, "mobile channel")
        add_reason(channel_lower.eq("atm"), 1, "atm cash movement")

    if country_col and home_country_col and country_col in working.columns and home_country_col in working.columns:
        current_country = working[country_col].fillna("").astype(str).str.upper()
        home_country = working[home_country_col].fillna("").astype(str).str.upper()
        add_reason(
            (current_country != "") & (home_country != "") & (current_country != home_country),
            3,
            "cross-border transaction differs from customer home country",
        )

    if city_col and home_city_col and city_col in working.columns and home_city_col in working.columns:
        current_city = working[city_col].fillna("").astype(str).str.lower()
        home_city = working[home_city_col].fillna("").astype(str).str.lower()
        add_reason(
            (current_city != "") & (home_city != "") & (current_city != home_city),
            1,
            "transaction city differs from customer home city",
        )

    if merchant_risk_col and merchant_risk_col in working.columns:
        merchant_risk_lower = working[merchant_risk_col].fillna("").astype(str).str.lower()
        add_reason(merchant_risk_lower.eq("high"), 3, "merchant risk level is high")
        add_reason(merchant_risk_lower.eq("medium"), 1, "merchant risk level is medium")

    if merchant_category_col and merchant_category_col in working.columns:
        merchant_cat_lower = working[merchant_category_col].fillna("").astype(str).str.lower()
        add_reason(
            merchant_cat_lower.isin(["luxury", "travel", "electronics", "gambling"]),
            1,
            "merchant category is commonly watched for risky spend behavior",
        )

    if customer_col and customer_col in working.columns and "_customer_avg_amount" in working.columns:
        cust_avg = pd.to_numeric(working["_customer_avg_amount"], errors="coerce")
        add_reason(
            (amt.notna()) & (cust_avg.notna()) & (cust_avg > 0) & (amt >= cust_avg * 3),
            2,
            "transaction amount is at least 3x the customer's batch average",
        )

    working["_suspicion_reasons"] = (
        working["_suspicion_reasons"]
        .astype(str)
        .str.strip()
        .replace("", "no major real-time fraud rules were triggered")
    )

    def bucket(score):
        if pd.isna(score):
            return "Unknown"
        if score >= 7:
            return "High"
        if score >= 4:
            return "Medium"
        if score >= 1:
            return "Low"
        return "Minimal"

    working["_suspicion_bucket"] = working["_suspicion_score"].apply(bucket)

    try:
        working = working.sort_values(
            by=["_suspicion_score", "_amount_numeric"],
            ascending=[False, False],
            na_position="last",
        )
    except Exception:
        pass

    return working


def get_realtime_fraud_summary() -> str:
    ctx = _prepare_streaming_context()
    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    scored = _score_streaming_suspicion(ctx["df"], ctx)

    high_df = scored[scored["_suspicion_bucket"] == "High"]
    med_df = scored[scored["_suspicion_bucket"] == "Medium"]
    low_df = scored[scored["_suspicion_bucket"] == "Low"]

    lines = [
        make_section_title("Real-Time Suspicious Activity Overview"),
    ]
    if ctx["source_path"] is not None:
        lines.append(make_kv_line("Source file", ctx["source_path"].name))

    lines.append(make_kv_line("Records evaluated", format_number(len(scored))))
    lines.append(make_kv_line("High Suspicion", format_number(len(high_df))))
    lines.append(make_kv_line("Medium Suspicion", format_number(len(med_df))))
    lines.append(make_kv_line("Low Suspicion", format_number(len(low_df))))

    try:
        lines.append(make_kv_line("Average suspicion score", f"{scored['_suspicion_score'].mean():.2f}"))
        lines.append(make_kv_line("Maximum suspicion score", f"{scored['_suspicion_score'].max():.2f}"))
    except Exception:
        pass

    return "\n".join(lines)


def get_top_suspicious_streaming_transactions(limit: int = 5, channel_filter: str | None = None) -> str:
    ctx = _prepare_streaming_context()
    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    scored = _score_streaming_suspicion(ctx["df"], ctx)

    if channel_filter and ctx["channel_col"] and ctx["channel_col"] in scored.columns:
        scored = scored[scored[ctx["channel_col"]].fillna("").astype(str).str.upper() == channel_filter.upper()]

    scored = scored.head(limit)

    if scored.empty:
        if channel_filter:
            return make_empty_message(f"No suspicious {channel_filter.upper()} transactions were found.")
        return make_empty_message("No suspicious streaming transactions were found.")

    title = f"Top {limit} Suspicious Streaming Transactions"
    if channel_filter:
        title = f"Top {limit} Suspicious {channel_filter.upper()} Transactions"

    lines = [make_section_title(title)]
    for _, row in scored.iterrows():
        extra_parts = [
            f"Suspicion Score: {row['_suspicion_score']}",
            f"Bucket: {_clean_value(row['_suspicion_bucket'])}",
            f"Reason: {_clean_value(row['_suspicion_reasons'])}",
        ]
        lines.append(
            _build_transaction_block(
                row=row,
                txn_col=ctx["txn_col"],
                customer_col=ctx["customer_col"],
                amount_col=ctx["amount_col"],
                channel_col=ctx["channel_col"],
                status_col=ctx["status_col"],
                timestamp_col=ctx["timestamp_col"],
                merchant_category_col=ctx["merchant_category_col"],
                country_col=ctx["country_col"],
                extra_parts=extra_parts,
            )
        )

    return "\n".join(lines)


def get_high_value_streaming_transactions(
    threshold: float = 10000,
    limit: int = 10,
    channel_filter: str | None = None,
) -> str:
    ctx = _prepare_streaming_context()
    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    df = ctx["df"]

    if channel_filter and ctx["channel_col"] and ctx["channel_col"] in df.columns:
        df = df[df[ctx["channel_col"]].fillna("").astype(str).str.upper() == channel_filter.upper()]

    filtered = df[df["_amount_numeric"] >= threshold].copy()

    if filtered.empty:
        if channel_filter:
            return make_empty_message(f"No {channel_filter.upper()} transactions above {threshold:,.2f} were found.")
        return make_empty_message(f"No streamed transactions above {threshold:,.2f} were found.")

    try:
        filtered = filtered.sort_values("_amount_numeric", ascending=False, na_position="last")
    except Exception:
        pass

    title = f"High-Value Streaming Transactions (Amount >= {threshold:,.2f})"
    if channel_filter:
        title = f"High-Value {channel_filter.upper()} Transactions (Amount >= {threshold:,.2f})"

    lines = [make_section_title(title)]
    for _, row in filtered.head(limit).iterrows():
        lines.append(
            _build_transaction_block(
                row=row,
                txn_col=ctx["txn_col"],
                customer_col=ctx["customer_col"],
                amount_col=ctx["amount_col"],
                channel_col=ctx["channel_col"],
                status_col=ctx["status_col"],
                timestamp_col=ctx["timestamp_col"],
                merchant_category_col=ctx["merchant_category_col"],
                country_col=ctx["country_col"],
            )
        )

    return "\n".join(lines)


def get_streaming_status_summary() -> str:
    ctx = _prepare_streaming_context()
    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    items = _top_counts(ctx["df"], ctx["status_col"], n=10)
    if not items:
        return make_empty_message("Status information is not available in the streaming data.")

    lines = [make_section_title("Streaming Status Breakdown")]
    for value, count in items:
        lines.append(f"- **{value}**: {format_number(count)}")
    return "\n".join(lines)


def get_top_streaming_customers(limit: int = 10) -> str:
    ctx = _prepare_streaming_context()
    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    df = ctx["df"]
    customer_col = ctx["customer_col"]

    if not customer_col or customer_col not in df.columns:
        return make_empty_message("Customer information is not available in the streaming data.")

    grouped = (
        df.groupby(customer_col, dropna=False)
        .agg(
            transactions=(customer_col, "size"),
            total_amount=("_amount_numeric", "sum"),
            avg_amount=("_amount_numeric", "mean"),
        )
        .reset_index()
        .sort_values(["total_amount", "transactions"], ascending=[False, False])
        .head(limit)
    )

    lines = [make_section_title(f"Top {limit} Customers in Latest Streaming Batch")]
    for _, row in grouped.iterrows():
        customer_value = _clean_value(row[customer_col])
        total_text = _format_amount(row["total_amount"])
        avg_text = _format_amount(row["avg_amount"])

        lines.append(
            f"- **{customer_value}** | Transactions: {format_number(row['transactions'])} | "
            f"Total Amount: {total_text} | Avg Amount: {avg_text}"
        )

    return "\n".join(lines)


def get_streaming_channel_summary() -> str:
    ctx = _prepare_streaming_context()
    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    df = ctx["df"]
    channel_col = ctx["channel_col"]

    if not channel_col or channel_col not in df.columns:
        return make_empty_message("Channel information is not available in the streaming data.")

    grouped = (
        df.groupby(channel_col, dropna=False)
        .agg(
            records=(channel_col, "size"),
            total_amount=("_amount_numeric", "sum"),
            avg_amount=("_amount_numeric", "mean"),
        )
        .reset_index()
        .sort_values(["records", "total_amount"], ascending=[False, False])
        .head(10)
    )

    lines = [make_section_title("Streaming Channel Summary")]
    for _, row in grouped.iterrows():
        channel_value = _clean_value(row[channel_col])
        total_text = _format_amount(row["total_amount"])
        avg_text = _format_amount(row["avg_amount"])

        lines.append(
            f"- **{channel_value}** | Records: {format_number(row['records'])} | "
            f"Total Amount: {total_text} | Avg Amount: {avg_text}"
        )

    return "\n".join(lines)


def get_streaming_country_summary() -> str:
    ctx = _prepare_streaming_context()
    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    df = ctx["df"]
    country_col = ctx["country_col"]

    if not country_col or country_col not in df.columns:
        return make_empty_message("Country information is not available in the streaming data.")

    grouped = (
        df.groupby(country_col, dropna=False)
        .agg(
            records=(country_col, "size"),
            total_amount=("_amount_numeric", "sum"),
            avg_amount=("_amount_numeric", "mean"),
        )
        .reset_index()
        .sort_values(["records", "total_amount"], ascending=[False, False])
        .head(10)
    )

    lines = [make_section_title("Streaming Country Summary")]
    for _, row in grouped.iterrows():
        country_value = _clean_value(row[country_col])
        total_text = _format_amount(row["total_amount"])
        avg_text = _format_amount(row["avg_amount"])

        lines.append(
            f"- **{country_value}** | Records: {format_number(row['records'])} | "
            f"Total Amount: {total_text} | Avg Amount: {avg_text}"
        )

    return "\n".join(lines)


def get_streaming_merchant_summary() -> str:
    ctx = _prepare_streaming_context()
    if ctx is None:
        return make_empty_message("Streaming data is unavailable.")

    df = ctx["df"]
    merchant_category_col = ctx["merchant_category_col"]

    if not merchant_category_col or merchant_category_col not in df.columns:
        return make_empty_message("Merchant category information is not available in the streaming data.")

    grouped = (
        df.groupby(merchant_category_col, dropna=False)
        .agg(
            records=(merchant_category_col, "size"),
            total_amount=("_amount_numeric", "sum"),
            avg_amount=("_amount_numeric", "mean"),
        )
        .reset_index()
        .sort_values(["records", "total_amount"], ascending=[False, False])
        .head(10)
    )

    lines = [make_section_title("Streaming Merchant Category Summary")]
    for _, row in grouped.iterrows():
        merchant_value = _clean_value(row[merchant_category_col])
        total_text = _format_amount(row["total_amount"])
        avg_text = _format_amount(row["avg_amount"])

        lines.append(
            f"- **{merchant_value}** | Records: {format_number(row['records'])} | "
            f"Total Amount: {total_text} | Avg Amount: {avg_text}"
        )

    return "\n".join(lines)


# =========================================================
# ROUTER
# =========================================================
def answer_streaming_question(user_query: str) -> str:
    q = user_query.lower().strip()
    limit = _extract_limit_from_query(q, default=5, max_limit=25)
    threshold = _extract_amount_threshold(q, default=10000.0)

    channel_filter = None
    if "atm" in q:
        channel_filter = "ATM"
    elif "pos" in q:
        channel_filter = "POS"
    elif "ecom" in q:
        channel_filter = "ECOM"
    elif "mobile" in q:
        channel_filter = "MOBILE"

    if not q:
        return ""

    if _contains_any(q, [
        "stream summary",
        "streaming summary",
        "stream overview",
        "streaming overview",
        "show streaming data",
        "show me streaming",
        "live feed",
        "transaction feed",
        "is streaming working",
        "pipeline active",
    ]):
        return get_streaming_summary()

    if _contains_any(q, [
        "latest timestamp",
        "latest time",
        "most recent timestamp",
        "newest timestamp",
        "last timestamp",
    ]):
        return get_latest_timestamp()

    if _contains_any(q, [
        "fraud summary",
        "real time fraud",
        "realtime fraud",
        "suspicious streaming",
        "high risk streaming transactions",
        "stream fraud",
        "streaming fraud",
        "fraud in stream",
    ]):
        return get_realtime_fraud_summary()

    if _contains_any(q, [
        "top suspicious streaming transactions",
        "top suspicious transactions",
        "most suspicious streaming transactions",
        "recent suspicious transactions",
        "latest suspicious transactions",
        "show suspicious transactions from latest stream",
        "suspicious atm transactions",
        "suspicious pos transactions",
        "suspicious ecom transactions",
        "suspicious mobile transactions",
    ]):
        return get_top_suspicious_streaming_transactions(limit=limit, channel_filter=channel_filter)

    if _contains_any(q, [
        "high value transactions",
        "large streamed transactions",
        "transactions above",
        "transactions over",
        "large transactions in stream",
        "latest high value transactions",
        "recent high value transactions",
        "big transactions",
    ]):
        return get_high_value_streaming_transactions(
            threshold=threshold,
            limit=limit,
            channel_filter=channel_filter,
        )

    if _contains_any(q, [
        "top customers in stream",
        "top streaming customers",
        "which customers spent the most",
        "streaming top customers",
    ]):
        return get_top_streaming_customers(limit=limit)

    if _contains_any(q, [
        "status breakdown",
        "streaming status",
        "approved vs declined",
        "declined transactions",
    ]):
        return get_streaming_status_summary()

    if _contains_any(q, [
        "channel summary",
        "streaming channels",
        "stream by channel",
        "top channels in stream",
    ]):
        return get_streaming_channel_summary()

    if _contains_any(q, [
        "country summary",
        "stream by country",
        "top countries in stream",
        "streaming countries",
    ]):
        return get_streaming_country_summary()

    if _contains_any(q, [
        "merchant summary",
        "merchant categories in stream",
        "stream by merchant",
        "top merchant categories in stream",
    ]):
        return get_streaming_merchant_summary()

    if _contains_any(q, [
        "last",
        "latest transactions",
        "recent transactions",
        "latest transaction",
        "show latest transactions",
        "show recent transactions",
        "last transactions",
    ]):
        return get_latest_transactions(limit=limit, channel_filter=channel_filter)

    return ""
from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd

from agentic_ai.utils.data_access import safe_load_parquet, safe_load_csv, standardize
from agentic_ai.utils.formatter import format_number
from agentic_ai.utils.presentation_formatter import (
    make_section_title,
    make_kv_line,
    make_bullet_list,
    make_empty_message,
)


# ---------------------------------------------------------
# FILE LOADING
# ---------------------------------------------------------
def _get_kpi_paths():
    """
    Import config lazily to avoid circular import issues.
    Returns candidate KPI file paths in priority order.
    """
    from agentic_ai import config

    candidate_paths = []

    if hasattr(config, "PREFERRED_PATHS") and "executive_kpis" in config.PREFERRED_PATHS:
        preferred = config.PREFERRED_PATHS["executive_kpis"]

        if isinstance(preferred, (list, tuple)):
            candidate_paths.extend(preferred)
        elif preferred is not None:
            candidate_paths.append(preferred)

    if hasattr(config, "DATA_PATHS"):
        data_paths = config.DATA_PATHS
        candidate_paths.extend([
            data_paths.get("executive_kpis_parquet"),
            data_paths.get("executive_kpis_csv"),
            data_paths.get("kpi_outputs"),
            data_paths.get("kpi_summary_csv"),
            data_paths.get("segment_summary_parquet"),
            data_paths.get("segment_summary_csv"),
            data_paths.get("channel_summary_parquet"),
            data_paths.get("channel_summary_csv"),
            data_paths.get("merchant_summary_parquet"),
            data_paths.get("merchant_summary_csv"),
            data_paths.get("geo_summary_parquet"),
            data_paths.get("geo_summary_csv"),
            data_paths.get("spend_prediction_summary_parquet"),
            data_paths.get("spend_prediction_summary_csv"),
        ])

    normalized = []
    seen = set()

    for path in candidate_paths:
        if path is None:
            continue

        try:
            p = Path(path)
        except Exception:
            continue

        key = str(p)
        if key not in seen:
            seen.add(key)
            normalized.append(p)

    return normalized


def _load_path(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or not path.is_file():
        return None

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return safe_load_parquet(path)
    if suffix == ".csv":
        return safe_load_csv(path)

    return None


def _load_best_kpi_dataset():
    """
    Load the first valid non-empty KPI dataset.
    """
    for path in _get_kpi_paths():
        df = _load_path(path)
        if df is not None and not df.empty:
            return standardize(df), path

    return None, None


def _load_all_kpi_datasets() -> list[tuple[pd.DataFrame, Path]]:
    """
    Load all available non-empty KPI datasets.
    This helps route questions to the most suitable grouped summary file.
    """
    datasets = []

    for path in _get_kpi_paths():
        df = _load_path(path)
        if df is not None and not df.empty:
            datasets.append((standardize(df), path))

    return datasets


# ---------------------------------------------------------
# STREAMING SUPPLEMENT
# ---------------------------------------------------------
def _load_live_stream_snapshot():
    """
    Import lazily to avoid circular imports and use the latest streaming batch
    as a live supplement to the KPI snapshot.
    """
    try:
        from agentic_ai.tools.streaming_tool import _prepare_streaming_context  # noqa: WPS433
    except Exception:
        return None

    try:
        ctx = _prepare_streaming_context()
    except Exception:
        return None

    if not ctx:
        return None

    df = ctx.get("df")
    if df is None or df.empty:
        return None

    return ctx


def _build_live_stream_kpi_lines(ctx: dict) -> list[str]:
    if not ctx:
        return []

    df = ctx["df"]
    source_path = ctx.get("source_path")
    timestamp_col = ctx.get("timestamp_col")
    customer_col = ctx.get("customer_col")
    amount_col = ctx.get("amount_col")
    channel_col = ctx.get("channel_col")
    country_col = ctx.get("country_col")
    customer_segment_col = ctx.get("customer_segment_col")

    lines = [make_section_title("Live Streaming Supplement")]

    if source_path is not None:
        try:
            lines.append(make_kv_line("Latest stream source", source_path.name))
        except Exception:
            lines.append(make_kv_line("Latest stream source", str(source_path)))

    lines.append(make_kv_line("Records in latest stream batch", format_number(len(df))))

    if customer_col and customer_col in df.columns:
        try:
            lines.append(
                make_kv_line(
                    "Unique customers in latest stream batch",
                    format_number(df[customer_col].nunique(dropna=True)),
                )
            )
        except Exception:
            pass

    amount_series = None
    if "_amount_numeric" in df.columns:
        amount_series = pd.to_numeric(df["_amount_numeric"], errors="coerce").dropna()
    elif amount_col and amount_col in df.columns:
        amount_series = pd.to_numeric(df[amount_col], errors="coerce").dropna()

    if amount_series is not None and not amount_series.empty:
        lines.append(make_kv_line("Stream batch total amount", f"{amount_series.sum():,.2f}"))
        lines.append(make_kv_line("Stream batch average amount", f"{amount_series.mean():,.2f}"))
        lines.append(make_kv_line("Stream batch max amount", f"{amount_series.max():,.2f}"))

    if timestamp_col and timestamp_col in df.columns:
        try:
            ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
            if not ts.empty:
                lines.append(make_kv_line("Latest stream timestamp", str(ts.max())))
                lines.append(make_kv_line("Earliest stream timestamp", str(ts.min())))
        except Exception:
            pass

    if channel_col and channel_col in df.columns:
        try:
            counts = df[channel_col].fillna("Unknown").astype(str).value_counts().head(3)
            if not counts.empty:
                items = [f"{name}: {format_number(count)}" for name, count in counts.items()]
                lines.append(make_bullet_list("Top stream channels", items))
        except Exception:
            pass

    if country_col and country_col in df.columns:
        try:
            counts = df[country_col].fillna("Unknown").astype(str).value_counts().head(3)
            if not counts.empty:
                items = [f"{name}: {format_number(count)}" for name, count in counts.items()]
                lines.append(make_bullet_list("Top stream countries", items))
        except Exception:
            pass

    if customer_segment_col and customer_segment_col in df.columns:
        try:
            counts = df[customer_segment_col].fillna("Unknown").astype(str).value_counts().head(3)
            if not counts.empty:
                items = [f"{name}: {format_number(count)}" for name, count in counts.items()]
                lines.append(make_bullet_list("Top stream customer segments", items))
        except Exception:
            pass

    return lines


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _pretty(col: str) -> str:
    return col.replace("_", " ").strip().title()


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _extract_limit_from_query(q: str, default: int = 5, max_limit: int = 25) -> int:
    match = re.search(r"\b(top|last|show|compare)\s+(\d+)\b", q)
    if match:
        try:
            value = int(match.group(2))
            return max(1, min(value, max_limit))
        except Exception:
            return default
    return default


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _sum_metric(df: pd.DataFrame, col: str) -> Optional[float]:
    if not col or col not in df.columns:
        return None
    series = _safe_numeric_series(df, col).dropna()
    if series.empty:
        return None
    return float(series.sum())


def _mean_metric(df: pd.DataFrame, col: str) -> Optional[float]:
    if not col or col not in df.columns:
        return None
    series = _safe_numeric_series(df, col).dropna()
    if series.empty:
        return None
    return float(series.mean())


def _max_metric(df: pd.DataFrame, col: str) -> Optional[float]:
    if not col or col not in df.columns:
        return None
    series = _safe_numeric_series(df, col).dropna()
    if series.empty:
        return None
    return float(series.max())


def _group_summary(df: pd.DataFrame, group_col: str, value_col: str, agg: str = "sum", top_n: int = 5):
    if group_col not in df.columns or value_col not in df.columns:
        return None

    temp = df[[group_col, value_col]].copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[value_col])

    if temp.empty:
        return None

    if agg == "mean":
        grouped = temp.groupby(group_col, dropna=False)[value_col].mean()
    elif agg == "max":
        grouped = temp.groupby(group_col, dropna=False)[value_col].max()
    else:
        grouped = temp.groupby(group_col, dropna=False)[value_col].sum()

    grouped = grouped.sort_values(ascending=False).head(top_n)
    return grouped


def _format_grouped_result(title: str, grouped) -> str:
    if grouped is None or len(grouped) == 0:
        return make_empty_message(f"{title} - No valid grouped data found.")

    items = []
    for idx, (name, val) in enumerate(grouped.items(), start=1):
        label = str(name) if pd.notna(name) else "Unknown"
        items.append(f"{idx}. {label}: {format_number(val)}")

    return "\n".join([
        make_section_title(title),
        make_bullet_list("Results", items),
    ])


def _compare_group(df: pd.DataFrame, group_col: str, value_col: str, agg: str = "sum") -> str:
    if group_col not in df.columns or value_col not in df.columns:
        return make_empty_message(f"Unable to compare {_pretty(group_col)} using {_pretty(value_col)}.")

    temp = df[[group_col, value_col]].copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[value_col])

    if temp.empty:
        return make_empty_message(f"No valid comparison data found for {_pretty(group_col)}.")

    if agg == "mean":
        grouped = temp.groupby(group_col, dropna=False)[value_col].mean().sort_values(ascending=False)
    elif agg == "max":
        grouped = temp.groupby(group_col, dropna=False)[value_col].max().sort_values(ascending=False)
    else:
        grouped = temp.groupby(group_col, dropna=False)[value_col].sum().sort_values(ascending=False)

    items = []
    for name, val in grouped.items():
        label = str(name) if pd.notna(name) else "Unknown"
        items.append(f"{label}: {format_number(val)}")

    return "\n".join([
        make_section_title(f"{_pretty(group_col)} Comparison"),
        make_bullet_list(f"By {_pretty(value_col)}", items),
    ])


def _detect_columns(df: pd.DataFrame) -> dict:
    return {
        "spend_col": _find_first_existing_column(df, ["total_spend", "spend", "revenue", "total_amount"]),
        "txn_col": _find_first_existing_column(df, ["total_transactions", "transactions", "transaction_count", "txn_count"]),
        "cust_col": _find_first_existing_column(df, ["total_customers", "customers", "customer_count", "unique_customers"]),
        "avg_col": _find_first_existing_column(df, ["avg_transaction_value", "average_transaction_value", "avg_amount", "average_amount"]),
        "segment_col": _find_first_existing_column(df, ["segment", "customer_segment", "segment_name"]),
        "channel_col": _find_first_existing_column(df, ["channel", "transaction_channel"]),
        "merchant_col": _find_first_existing_column(df, ["merchant_category", "category", "merchant"]),
        "city_col": _find_first_existing_column(df, ["city", "customer_city", "txn_city"]),
        "country_col": _find_first_existing_column(df, ["country", "customer_country", "txn_country"]),
    }


def _dataset_score_for_query(df: pd.DataFrame, q: str) -> int:
    cols = _detect_columns(df)
    score = 0

    if _contains_any(q, ["segment"]) and cols["segment_col"]:
        score += 3
    if _contains_any(q, ["channel", "atm", "pos", "ecom", "mobile"]) and cols["channel_col"]:
        score += 3
    if _contains_any(q, ["merchant", "category"]) and cols["merchant_col"]:
        score += 3
    if _contains_any(q, ["city"]) and cols["city_col"]:
        score += 3
    if _contains_any(q, ["country"]) and cols["country_col"]:
        score += 3

    if _contains_any(q, ["spend", "revenue", "money processed"]) and cols["spend_col"]:
        score += 2
    if _contains_any(q, ["transactions", "transaction count", "volume"]) and cols["txn_col"]:
        score += 2
    if _contains_any(q, ["customers", "customer count"]) and cols["cust_col"]:
        score += 2
    if _contains_any(q, ["average amount", "avg amount", "average transaction amount"]) and cols["avg_col"]:
        score += 2

    if len(df) == 1:
        score += 1

    return score


def _select_best_dataset_for_query(user_query: str):
    q = user_query.lower().strip()
    datasets = _load_all_kpi_datasets()

    if not datasets:
        return None, None

    best_df = None
    best_path = None
    best_score = -1

    for df, path in datasets:
        score = _dataset_score_for_query(df, q)
        if score > best_score:
            best_score = score
            best_df = df
            best_path = path

    return best_df, best_path


# ---------------------------------------------------------
# KPI SUMMARY
# ---------------------------------------------------------
def get_kpi_summary() -> str:
    df, source_path = _load_best_kpi_dataset()

    if df is None:
        stream_ctx = _load_live_stream_snapshot()
        if stream_ctx:
            lines = [
                make_empty_message("KPI data is unavailable."),
                make_section_title("Available Live View"),
            ]
            lines.extend(_build_live_stream_kpi_lines(stream_ctx))
            return "\n".join(lines)
        return make_empty_message("KPI data is unavailable.")

    cols = _detect_columns(df)

    total_customers_col = cols["cust_col"]
    total_transactions_col = cols["txn_col"]
    total_spend_col = cols["spend_col"]
    avg_transaction_value_col = cols["avg_col"]

    stream_ctx = _load_live_stream_snapshot()

    if len(df) == 1:
        row = df.iloc[0]
        lines = [make_section_title("Latest KPI Summary")]

        if source_path is not None:
            lines.append(make_kv_line("Source file", source_path.name))

        if total_customers_col:
            value = _safe_float(row[total_customers_col])
            if value is not None:
                lines.append(make_kv_line("Total customers", f"{value:,.0f}"))

        if total_transactions_col:
            value = _safe_float(row[total_transactions_col])
            if value is not None:
                lines.append(make_kv_line("Total transactions", f"{value:,.0f}"))

        if total_spend_col:
            value = _safe_float(row[total_spend_col])
            if value is not None:
                lines.append(make_kv_line("Total spend", f"{value:,.2f}"))

        if avg_transaction_value_col:
            value = _safe_float(row[avg_transaction_value_col])
            if value is not None:
                lines.append(make_kv_line("Average transaction value", f"{value:,.2f}"))

        if stream_ctx:
            lines.extend(_build_live_stream_kpi_lines(stream_ctx))

        return "\n".join(lines)

    lines = [
        make_section_title("KPI Summary"),
        make_kv_line("Source file", source_path.name if source_path is not None else "Unknown"),
        make_kv_line("Rows", format_number(len(df))),
    ]

    if total_customers_col:
        series = pd.to_numeric(df[total_customers_col], errors="coerce").dropna()
        if not series.empty:
            lines.append(make_kv_line("Average customers across rows", f"{series.mean():,.0f}"))
            lines.append(make_kv_line("Max customers in a row", f"{series.max():,.0f}"))

    if total_transactions_col:
        series = pd.to_numeric(df[total_transactions_col], errors="coerce").dropna()
        if not series.empty:
            lines.append(make_kv_line("Total transactions across rows", f"{series.sum():,.0f}"))
            lines.append(make_kv_line("Average transactions per row", f"{series.mean():,.2f}"))

    if total_spend_col:
        series = pd.to_numeric(df[total_spend_col], errors="coerce").dropna()
        if not series.empty:
            lines.append(make_kv_line("Total spend across rows", f"{series.sum():,.2f}"))
            lines.append(make_kv_line("Average spend per row", f"{series.mean():,.2f}"))
            lines.append(make_kv_line("Max spend in a row", f"{series.max():,.2f}"))

    if avg_transaction_value_col:
        series = pd.to_numeric(df[avg_transaction_value_col], errors="coerce").dropna()
        if not series.empty:
            lines.append(make_kv_line("Average transaction value", f"{series.mean():,.2f}"))

    if stream_ctx:
        lines.extend(_build_live_stream_kpi_lines(stream_ctx))

    return "\n".join(lines)


# ---------------------------------------------------------
# KPI QUESTION ANSWERING
# ---------------------------------------------------------
def answer_kpi_question(user_query: str) -> str:
    q = user_query.lower().strip()
    top_n = _extract_limit_from_query(q, default=5, max_limit=25)

    if _contains_any(q, [
        "kpi summary",
        "show kpi summary",
        "latest kpi",
        "latest kpis",
        "executive summary",
        "business summary",
        "dashboard summary",
        "overview",
    ]):
        return get_kpi_summary()

    if _contains_any(q, [
        "live kpi",
        "live kpis",
        "stream kpi",
        "streaming kpi",
        "latest stream kpi",
        "latest live kpi",
        "latest stream summary",
    ]):
        stream_ctx = _load_live_stream_snapshot()
        if stream_ctx:
            return "\n".join([make_section_title("Live KPI View")] + _build_live_stream_kpi_lines(stream_ctx))
        return make_empty_message("Live streaming KPI data is unavailable.")

    df, source_path = _select_best_dataset_for_query(user_query)

    if df is None:
        stream_ctx = _load_live_stream_snapshot()
        if stream_ctx:
            return "\n".join(
                [
                    make_empty_message("Batch KPI data is unavailable."),
                    make_section_title("Live KPI View"),
                ] + _build_live_stream_kpi_lines(stream_ctx)
            )
        return make_empty_message("KPI data is unavailable.")

    cols = _detect_columns(df)

    spend_col = cols["spend_col"]
    txn_col = cols["txn_col"]
    cust_col = cols["cust_col"]
    avg_col = cols["avg_col"]
    segment_col = cols["segment_col"]
    channel_col = cols["channel_col"]
    merchant_col = cols["merchant_col"]
    city_col = cols["city_col"]
    country_col = cols["country_col"]

    source_name = source_path.name if source_path is not None else "Unknown"

    if _contains_any(q, ["average transaction amount", "avg amount", "average amount"]):
        if avg_col:
            value = _mean_metric(df, avg_col)
            if value is not None:
                return "\n".join([
                    make_section_title("Average Transaction Amount"),
                    make_kv_line("Source", source_name),
                    make_kv_line("Value", f"{value:,.2f}"),
                ])
        return make_empty_message("Average transaction amount is unavailable.")

    if _contains_any(q, ["total spend", "revenue", "money processed", "processed money", "total amount"]):
        if spend_col:
            value = _sum_metric(df, spend_col)
            if value is not None:
                return "\n".join([
                    make_section_title("Total Spend"),
                    make_kv_line("Source", source_name),
                    make_kv_line("Value", f"{value:,.2f}"),
                ])
        return make_empty_message("Total spend is unavailable.")

    if _contains_any(q, ["transaction count", "total transactions", "how many transactions", "volume"]):
        if txn_col:
            value = _sum_metric(df, txn_col)
            if value is not None:
                return "\n".join([
                    make_section_title("Transaction Count"),
                    make_kv_line("Source", source_name),
                    make_kv_line("Value", f"{value:,.0f}"),
                ])
        return make_empty_message("Transaction count is unavailable.")

    if _contains_any(q, ["how many customers", "total customers", "customer count"]):
        if cust_col:
            value = _sum_metric(df, cust_col)
            if value is not None:
                return "\n".join([
                    make_section_title("Customer Count"),
                    make_kv_line("Source", source_name),
                    make_kv_line("Value", f"{value:,.0f}"),
                ])
        return make_empty_message("Customer count is unavailable.")

    if "segment" in q:
        if _contains_any(q, ["highest spend", "top segment", "which segment", "best segment"]):
            if segment_col and spend_col:
                grouped = _group_summary(df, segment_col, spend_col, agg="sum", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Segments by Spend from {source_name}", grouped)
        if _contains_any(q, ["average spend", "avg spend"]):
            if segment_col and spend_col:
                grouped = _group_summary(df, segment_col, spend_col, agg="mean", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Segments by Average Spend from {source_name}", grouped)
        if _contains_any(q, ["compare", "performance", "segment wise", "by segment"]):
            if segment_col and spend_col:
                return _compare_group(df, segment_col, spend_col)

    if _contains_any(q, ["channel", "pos", "atm", "ecom", "mobile"]):
        if _contains_any(q, ["highest", "top", "which channel", "best channel"]):
            if channel_col and spend_col:
                grouped = _group_summary(df, channel_col, spend_col, agg="sum", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Channels by Spend from {source_name}", grouped)
        if _contains_any(q, ["average", "avg"]):
            if channel_col and spend_col:
                grouped = _group_summary(df, channel_col, spend_col, agg="mean", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Channels by Average Spend from {source_name}", grouped)
        if _contains_any(q, ["compare", "vs", "performance", "by channel"]):
            if channel_col and spend_col:
                return _compare_group(df, channel_col, spend_col)

    if _contains_any(q, ["merchant", "category", "merchant category"]):
        if _contains_any(q, ["top", "highest", "best"]):
            if merchant_col and spend_col:
                grouped = _group_summary(df, merchant_col, spend_col, agg="sum", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Merchant Categories by Spend from {source_name}", grouped)
        if _contains_any(q, ["average", "avg"]):
            if merchant_col and spend_col:
                grouped = _group_summary(df, merchant_col, spend_col, agg="mean", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Merchant Categories by Average Spend from {source_name}", grouped)
        if _contains_any(q, ["compare", "performance"]):
            if merchant_col and spend_col:
                return _compare_group(df, merchant_col, spend_col)

    if "city" in q:
        if _contains_any(q, ["highest", "top", "which city", "best city"]):
            if city_col and spend_col:
                grouped = _group_summary(df, city_col, spend_col, agg="sum", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Cities by Spend from {source_name}", grouped)
        if _contains_any(q, ["average", "avg"]):
            if city_col and spend_col:
                grouped = _group_summary(df, city_col, spend_col, agg="mean", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Cities by Average Spend from {source_name}", grouped)
        if _contains_any(q, ["compare", "performance", "by city"]):
            if city_col and spend_col:
                return _compare_group(df, city_col, spend_col)

    if "country" in q:
        if _contains_any(q, ["highest", "top", "which country", "best country"]):
            if country_col and spend_col:
                grouped = _group_summary(df, country_col, spend_col, agg="sum", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Countries by Spend from {source_name}", grouped)
        if _contains_any(q, ["average", "avg"]):
            if country_col and spend_col:
                grouped = _group_summary(df, country_col, spend_col, agg="mean", top_n=top_n)
                return _format_grouped_result(f"Top {top_n} Countries by Average Spend from {source_name}", grouped)
        if _contains_any(q, ["compare", "performance", "by country"]):
            if country_col and spend_col:
                return _compare_group(df, country_col, spend_col)

    if _contains_any(q, ["top channels"]) and channel_col and spend_col:
        grouped = _group_summary(df, channel_col, spend_col, agg="sum", top_n=top_n)
        return _format_grouped_result(f"Top {top_n} Channels by Spend from {source_name}", grouped)

    if _contains_any(q, ["top segments"]) and segment_col and spend_col:
        grouped = _group_summary(df, segment_col, spend_col, agg="sum", top_n=top_n)
        return _format_grouped_result(f"Top {top_n} Segments by Spend from {source_name}", grouped)

    if _contains_any(q, ["top cities"]) and city_col and spend_col:
        grouped = _group_summary(df, city_col, spend_col, agg="sum", top_n=top_n)
        return _format_grouped_result(f"Top {top_n} Cities by Spend from {source_name}", grouped)

    if _contains_any(q, ["top countries"]) and country_col and spend_col:
        grouped = _group_summary(df, country_col, spend_col, agg="sum", top_n=top_n)
        return _format_grouped_result(f"Top {top_n} Countries by Spend from {source_name}", grouped)

    if _contains_any(q, ["top merchants", "top merchant categories"]) and merchant_col and spend_col:
        grouped = _group_summary(df, merchant_col, spend_col, agg="sum", top_n=top_n)
        return _format_grouped_result(f"Top {top_n} Merchant Categories by Spend from {source_name}", grouped)

    # IMPORTANT:
    # Return empty string instead of full summary.
    # Let agent.py decide whether to fallback to summary.
    return ""
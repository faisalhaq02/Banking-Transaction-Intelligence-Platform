from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from agentic_ai.config import TOP_N_DEFAULT
from agentic_ai.utils.cloud_data_access import load_risk_cloud_first
from agentic_ai.utils.data_access import first_matching_column, standardize
from agentic_ai.utils.formatter import format_number
from agentic_ai.utils.presentation_formatter import (
    make_section_title,
    make_kv_line,
    make_bullet_list,
    make_empty_message,
)


# =========================================================
# BASIC HELPERS
# =========================================================
def _safe_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series(dtype="float64")


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _extract_limit_from_query(
    q: str,
    default: int = TOP_N_DEFAULT,
    max_limit: int = 50,
) -> int:
    match = re.search(r"\b(top|last|latest|show)\s+(\d+)\b", q)
    if match:
        try:
            value = int(match.group(2))
            return max(1, min(value, max_limit))
        except Exception:
            return default
    return default


def _extract_customer_id(user_query: str) -> str | None:
    match = re.search(r"\bC\d+\b", user_query.upper())
    if match:
        return match.group(0)
    return None


def _get_risk_bucket(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 0.85:
        return "High"
    if score >= 0.60:
        return "Medium"
    return "Low"


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


def _format_score(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.4f}"
    except Exception:
        return "—"


def _format_amount(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return format_number(float(value))
    except Exception:
        return "—"


def _add_if_present(parts: list[str], label: str, value) -> None:
    cleaned = _clean_value(value)
    if cleaned != "—":
        parts.append(f"{label}: {cleaned}")


def _top_counts(df: pd.DataFrame, col: str, n: int = 5):
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


def _format_top_counts(title: str, items) -> str | None:
    if not items:
        return None
    bullet_items = [f"{value}: {format_number(count)}" for value, count in items]
    return make_bullet_list(title, bullet_items)


# =========================================================
# BATCH / CLOUD RISK CONTEXT
# =========================================================
def _prepare_risk_context():
    df = load_risk_cloud_first()

    if df is None:
        return None

    if df.empty:
        return {"empty": True, "df": df}

    df = standardize(df)

    customer_col = first_matching_column(df, ["customer_id", "cust_id", "customer"])
    risk_col = first_matching_column(df, ["risk_score", "score", "probability", "fraud_risk_score"])
    segment_col = first_matching_column(df, ["customer_segment", "segment", "segment_name"])
    country_col = first_matching_column(df, ["country", "customer_country", "home_country"])
    channel_col = first_matching_column(df, ["channel", "txn_channel", "preferred_channel"])
    status_col = first_matching_column(df, ["status", "risk_status", "label"])
    reason_col = first_matching_column(df, ["risk_reason", "reason", "explanation", "risk_driver"])
    amount_col = first_matching_column(df, ["amount", "total_amount", "spend", "total_spend", "linked_amount"])
    timestamp_col = first_matching_column(df, ["timestamp", "event_time", "datetime", "created_at"])

    working_df = df.copy()

    if risk_col and risk_col in working_df.columns:
        working_df["_risk_numeric"] = _safe_numeric(working_df[risk_col])
        working_df["_risk_bucket"] = working_df["_risk_numeric"].apply(_get_risk_bucket)
    else:
        working_df["_risk_numeric"] = pd.Series([pd.NA] * len(working_df), index=working_df.index)
        working_df["_risk_bucket"] = "Unknown"

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

    sorted_df = working_df.copy()
    try:
        sorted_df = sorted_df.sort_values(
            by=["_risk_numeric", "_amount_numeric"],
            ascending=[False, False],
            na_position="last",
        )
    except Exception:
        pass

    return {
        "empty": False,
        "df": working_df,
        "sorted_df": sorted_df,
        "customer_col": customer_col,
        "risk_col": risk_col,
        "segment_col": segment_col,
        "country_col": country_col,
        "channel_col": channel_col,
        "status_col": status_col,
        "reason_col": reason_col,
        "amount_col": amount_col,
        "timestamp_col": timestamp_col,
    }


# =========================================================
# LIVE STREAM / CLOUD SUPPLEMENT
# =========================================================
def _load_live_stream_context():
    try:
        from agentic_ai.tools.streaming_tool import _prepare_streaming_context
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


def _score_live_risk(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    working = df.copy()

    amount_col = ctx.get("amount_col")
    status_col = ctx.get("status_col")
    channel_col = ctx.get("channel_col")
    country_col = ctx.get("country_col")
    home_country_col = ctx.get("home_country_col")
    city_col = ctx.get("city_col")
    home_city_col = ctx.get("home_city_col")
    merchant_risk_col = ctx.get("merchant_risk_col")
    merchant_category_col = ctx.get("merchant_category_col")
    customer_col = ctx.get("customer_col")

    working["_live_risk_score"] = 0.0
    working["_live_risk_reason"] = ""

    def add_reason(mask, points, reason_text):
        if mask is None:
            return
        try:
            mask = mask.fillna(False)
        except Exception:
            return

        working.loc[mask, "_live_risk_score"] += points
        working.loc[mask, "_live_risk_reason"] = (
            working.loc[mask, "_live_risk_reason"]
            .astype(str)
            .apply(lambda x: f"{x}; {reason_text}".strip("; ").strip())
        )

    amt = pd.to_numeric(working.get("_amount_numeric"), errors="coerce")

    if amount_col and "_amount_numeric" in working.columns:
        add_reason(amt >= 10000, 0.35, "very high transaction amount >= 10,000")
        add_reason((amt >= 5000) & (amt < 10000), 0.18, "high transaction amount between 5,000 and 9,999.99")

        try:
            batch_mean = amt.dropna().mean()
            batch_std = amt.dropna().std()
            if pd.notna(batch_mean) and pd.notna(batch_std) and batch_std > 0:
                add_reason(amt >= (batch_mean + 2 * batch_std), 0.15, "amount is well above batch normal range")
        except Exception:
            pass

    if status_col and status_col in working.columns:
        status_lower = working[status_col].fillna("").astype(str).str.lower()
        add_reason(status_lower.eq("declined"), 0.12, "transaction was declined")
        add_reason(status_lower.eq("pending"), 0.05, "transaction is pending")

    if channel_col and channel_col in working.columns:
        channel_lower = working[channel_col].fillna("").astype(str).str.lower()
        add_reason(channel_lower.eq("atm"), 0.05, "atm transaction")
        add_reason(channel_lower.eq("ecom"), 0.06, "ecommerce transaction")
        add_reason(channel_lower.eq("mobile"), 0.04, "mobile transaction")

    if country_col and home_country_col and country_col in working.columns and home_country_col in working.columns:
        current_country = working[country_col].fillna("").astype(str).str.upper()
        home_country = working[home_country_col].fillna("").astype(str).str.upper()
        add_reason(
            (current_country != "") & (home_country != "") & (current_country != home_country),
            0.20,
            "cross-border transaction differs from home country",
        )

    if city_col and home_city_col and city_col in working.columns and home_city_col in working.columns:
        current_city = working[city_col].fillna("").astype(str).str.lower()
        home_city = working[home_city_col].fillna("").astype(str).str.lower()
        add_reason(
            (current_city != "") & (home_city != "") & (current_city != home_city),
            0.06,
            "transaction city differs from home city",
        )

    if merchant_risk_col and merchant_risk_col in working.columns:
        merchant_risk_lower = working[merchant_risk_col].fillna("").astype(str).str.lower()
        add_reason(merchant_risk_lower.eq("high"), 0.22, "merchant risk level is high")
        add_reason(merchant_risk_lower.eq("medium"), 0.08, "merchant risk level is medium")

    if merchant_category_col and merchant_category_col in working.columns:
        merchant_cat_lower = working[merchant_category_col].fillna("").astype(str).str.lower()
        add_reason(
            merchant_cat_lower.isin(["luxury", "travel", "electronics", "gambling"]),
            0.07,
            "merchant category is commonly monitored",
        )

    if customer_col and customer_col in working.columns and "_customer_avg_amount" in working.columns:
        cust_avg = pd.to_numeric(working["_customer_avg_amount"], errors="coerce")
        add_reason(
            (amt.notna()) & (cust_avg.notna()) & (cust_avg > 0) & (amt >= cust_avg * 3),
            0.15,
            "transaction amount is at least 3x customer batch average",
        )

    working["_live_risk_score"] = working["_live_risk_score"].clip(lower=0, upper=1)

    def risk_bucket(score):
        if pd.isna(score):
            return "Unknown"
        if score >= 0.85:
            return "High"
        if score >= 0.60:
            return "Medium"
        return "Low"

    working["_live_risk_bucket"] = working["_live_risk_score"].apply(risk_bucket)
    working["_live_risk_reason"] = (
        working["_live_risk_reason"]
        .astype(str)
        .str.strip()
        .replace("", "no major live risk rules were triggered")
    )

    try:
        working = working.sort_values(
            by=["_live_risk_score", "_amount_numeric"],
            ascending=[False, False],
            na_position="last",
        )
    except Exception:
        pass

    return working


# =========================================================
# PRESENTATION HELPERS
# =========================================================
def _build_record_block(
    row,
    customer_col,
    risk_score_value,
    bucket_value,
    segment_col,
    country_col,
    channel_col,
    status_col,
    amount_col,
    reason_value,
    timestamp_col,
) -> str:
    title = _clean_value(row.get(customer_col)) if customer_col else "—"

    parts = []
    if risk_score_value != "—":
        parts.append(f"Risk Score: {risk_score_value}")
    if bucket_value != "—":
        parts.append(f"Bucket: {bucket_value}")
    if amount_col:
        amt = _format_amount(row.get(amount_col))
        if amt != "—":
            parts.append(f"Amount: {amt}")
    if segment_col:
        _add_if_present(parts, "Segment", row.get(segment_col))
    if country_col:
        _add_if_present(parts, "Country", row.get(country_col))
    if channel_col:
        _add_if_present(parts, "Channel", row.get(channel_col))
    if status_col:
        _add_if_present(parts, "Status", row.get(status_col))
    if timestamp_col:
        _add_if_present(parts, "Timestamp", row.get(timestamp_col))
    if reason_value != "—":
        parts.append(f"Reason: {reason_value}")

    return f"- **{title}**\n  " + " | ".join(parts)


def _format_customer_risk_lines(df: pd.DataFrame, ctx: dict, limit: int) -> list[str]:
    customer_col = ctx["customer_col"]
    risk_col = ctx["risk_col"]
    segment_col = ctx["segment_col"]
    country_col = ctx["country_col"]
    channel_col = ctx["channel_col"]
    status_col = ctx["status_col"]
    reason_col = ctx["reason_col"]
    amount_col = ctx["amount_col"]
    timestamp_col = ctx["timestamp_col"]

    lines = []
    for _, row in df.head(limit).iterrows():
        risk_value = _format_score(row[risk_col]) if risk_col and risk_col in row.index else "—"
        bucket_value = _clean_value(row.get("_risk_bucket"))
        reason_value = _clean_value(row.get(reason_col)) if reason_col else "—"

        lines.append(
            _build_record_block(
                row=row,
                customer_col=customer_col,
                risk_score_value=risk_value,
                bucket_value=bucket_value,
                segment_col=segment_col,
                country_col=country_col,
                channel_col=channel_col,
                status_col=status_col,
                amount_col=amount_col,
                reason_value=reason_value,
                timestamp_col=timestamp_col,
            )
        )
    return lines


def _build_live_risk_lines(ctx: dict, limit: int = 5) -> list[str]:
    df = ctx["df"]
    source_path = ctx.get("source_path")
    scored = _score_live_risk(df, ctx)

    customer_col = ctx.get("customer_col")
    amount_col = ctx.get("amount_col")
    channel_col = ctx.get("channel_col")
    country_col = ctx.get("country_col")
    status_col = ctx.get("status_col")
    segment_col = ctx.get("customer_segment_col")
    timestamp_col = ctx.get("timestamp_col")

    high_df = scored[scored["_live_risk_bucket"] == "High"]
    med_df = scored[scored["_live_risk_bucket"] == "Medium"]
    low_df = scored[scored["_live_risk_bucket"] == "Low"]

    lines = [make_section_title("Live Risk Snapshot")]

    if source_path is not None:
        try:
            lines.append(make_kv_line("Latest batch", source_path.name))
        except Exception:
            lines.append(make_kv_line("Latest batch", str(source_path)))

    lines.append(make_kv_line("Records evaluated", format_number(len(scored))))
    lines.append(make_kv_line("High Risk", format_number(len(high_df))))
    lines.append(make_kv_line("Medium Risk", format_number(len(med_df))))
    lines.append(make_kv_line("Low Risk", format_number(len(low_df))))

    try:
        lines.append(make_kv_line("Average live risk score", f"{scored['_live_risk_score'].mean():.4f}"))
        lines.append(make_kv_line("Max live risk score", f"{scored['_live_risk_score'].max():.4f}"))
    except Exception:
        pass

    lines.append(make_section_title("Top Live-Risk Records"))
    for _, row in scored.head(limit).iterrows():
        lines.append(
            _build_record_block(
                row=row,
                customer_col=customer_col,
                risk_score_value=_format_score(row.get("_live_risk_score")),
                bucket_value=_clean_value(row.get("_live_risk_bucket")),
                segment_col=segment_col,
                country_col=country_col,
                channel_col=channel_col,
                status_col=status_col,
                amount_col=amount_col,
                reason_value=_clean_value(row.get("_live_risk_reason")),
                timestamp_col=timestamp_col,
            )
        )

    return lines


def _generate_key_insight(df: pd.DataFrame, ctx: dict) -> str:
    if df is None or df.empty:
        return "No risk insight is available."

    segment_col = ctx.get("segment_col")
    amount_col = ctx.get("amount_col")

    insights = []

    if segment_col and segment_col in df.columns:
        high_df = df[df["_risk_bucket"] == "High"]
        if not high_df.empty:
            top_segment = high_df[segment_col].astype(str).value_counts().head(1)
            if not top_segment.empty:
                insights.append(f"High-risk exposure is concentrated in **segment {top_segment.index[0]}**.")

    if amount_col and "_amount_numeric" in df.columns:
        amt = df["_amount_numeric"].dropna()
        if not amt.empty:
            insights.append(f"The average linked amount across scored customers is **{format_number(amt.mean())}**, so even medium-risk cases may carry material exposure.")

    score_series = df["_risk_numeric"].dropna()
    if not score_series.empty:
        high_count = int((score_series >= 0.85).sum())
        if high_count > 0:
            insights.append(f"There are **{format_number(high_count)} high-risk customers** requiring priority review.")

    return insights[0] if insights else "Risk scoring is available, but richer detail fields would improve explainability."


# =========================================================
# PUBLIC SUMMARY
# =========================================================
def get_risk_summary(top_n: int = TOP_N_DEFAULT) -> str:
    ctx = _prepare_risk_context()
    live_ctx = _load_live_stream_context()

    if ctx is None and live_ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx is not None and ctx.get("empty"):
        if live_ctx:
            return "\n".join([
                make_section_title("Risk Overview"),
                make_empty_message("Risk score data is available but empty."),
            ] + _build_live_risk_lines(live_ctx, limit=top_n))
        return make_empty_message("Risk score data is available but empty.")

    if ctx is None:
        return "\n".join([
            make_section_title("Risk Overview"),
            make_empty_message("Batch/cloud risk data is unavailable."),
        ] + _build_live_risk_lines(live_ctx, limit=top_n))

    df = ctx["df"]
    sorted_df = ctx["sorted_df"]
    customer_col = ctx["customer_col"]
    risk_col = ctx["risk_col"]
    amount_col = ctx["amount_col"]
    timestamp_col = ctx["timestamp_col"]

    if risk_col is None:
        return make_empty_message(
            f"Risk score file exists, but no risk score column was found. Available columns: {list(df.columns)}"
        )

    lines = [
        make_section_title("Risk Overview"),
        make_section_title("Historical Summary"),
        make_kv_line("Records evaluated", format_number(len(df))),
    ]

    if customer_col and customer_col in df.columns:
        lines.append(make_kv_line("Unique customers", format_number(df[customer_col].nunique(dropna=True))))

    score_series = df["_risk_numeric"].dropna()
    if not score_series.empty:
        lines.append(make_kv_line("Average risk score", f"{score_series.mean():.4f}"))
        lines.append(make_kv_line("Max risk score", f"{score_series.max():.4f}"))
        lines.append(
            make_bullet_list(
                "Risk Distribution",
                [
                    f"High Risk: {format_number((score_series >= 0.85).sum())}",
                    f"Medium Risk: {format_number(((score_series >= 0.60) & (score_series < 0.85)).sum())}",
                    f"Low Risk: {format_number((score_series < 0.60).sum())}",
                ],
            )
        )

    if amount_col and amount_col in df.columns:
        amt = df["_amount_numeric"].dropna()
        if not amt.empty:
            lines.append(
                make_bullet_list(
                    "Exposure Summary",
                    [
                        f"Total linked amount: {format_number(amt.sum())}",
                        f"Average linked amount: {format_number(amt.mean())}",
                        f"Max linked amount: {format_number(amt.max())}",
                    ],
                )
            )

    if timestamp_col and timestamp_col in df.columns:
        ts = df["_parsed_timestamp"].dropna()
        if not ts.empty:
            lines.append(
                make_bullet_list(
                    "Time Range",
                    [
                        f"Earliest timestamp: {ts.min()}",
                        f"Latest timestamp: {ts.max()}",
                    ],
                )
            )

    segment_block = _format_top_counts("Top Segments", _top_counts(df, ctx["segment_col"]))
    country_block = _format_top_counts("Top Countries", _top_counts(df, ctx["country_col"]))
    channel_block = _format_top_counts("Top Channels", _top_counts(df, ctx["channel_col"]))

    if segment_block:
        lines.append(segment_block)
    if country_block:
        lines.append(country_block)
    if channel_block:
        lines.append(channel_block)

    lines.append(make_section_title(f"Top {min(top_n, len(sorted_df))} High-Risk Customers"))
    lines.extend(_format_customer_risk_lines(sorted_df, ctx, top_n))

    if live_ctx:
        lines.extend(_build_live_risk_lines(live_ctx, limit=top_n))

    lines.append(make_section_title("Key Insight"))
    lines.append(_generate_key_insight(df, ctx))

    return "\n".join(lines)


def get_top_risk_customers(top_n: int = TOP_N_DEFAULT) -> str:
    ctx = _prepare_risk_context()
    live_ctx = _load_live_stream_context()

    if ctx is None and live_ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx is None:
        return "\n".join([
            make_section_title("Top High-Risk Customers"),
            make_empty_message("Batch/cloud risk data is unavailable."),
        ] + _build_live_risk_lines(live_ctx, limit=top_n))

    if ctx["empty"]:
        return make_empty_message("Risk score data is available but empty.")

    if ctx["risk_col"] is None:
        return make_empty_message(
            f"Risk score file exists, but no risk score column was found. Available columns: {list(ctx['df'].columns)}"
        )

    lines = [make_section_title(f"Top {min(top_n, len(ctx['sorted_df']))} High-Risk Customers")]
    lines.extend(_format_customer_risk_lines(ctx["sorted_df"], ctx, top_n))

    if live_ctx:
        lines.extend(_build_live_risk_lines(live_ctx, limit=min(top_n, 5)))

    return "\n".join(lines)


def get_high_risk_summary(top_n: int = TOP_N_DEFAULT) -> str:
    ctx = _prepare_risk_context()
    live_ctx = _load_live_stream_context()

    if ctx is None and live_ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx is None:
        return "\n".join([
            make_section_title("High-Risk Customer Summary"),
            make_empty_message("Batch/cloud risk data is unavailable."),
        ] + _build_live_risk_lines(live_ctx, limit=top_n))

    if ctx["empty"]:
        return make_empty_message("Risk score data is available but empty.")

    df = ctx["df"]
    high_risk_df = df[df["_risk_bucket"] == "High"].copy()

    if high_risk_df.empty:
        if live_ctx:
            return "\n".join([
                make_section_title("High-Risk Customer Summary"),
                make_empty_message("No batch/cloud high-risk customers were found."),
            ] + _build_live_risk_lines(live_ctx, limit=top_n))
        return make_empty_message("No high-risk customers were found.")

    try:
        high_risk_df = high_risk_df.sort_values(
            by=["_risk_numeric", "_amount_numeric"],
            ascending=[False, False],
            na_position="last",
        )
    except Exception:
        pass

    lines = [
        make_section_title("High-Risk Customer Summary"),
        make_kv_line("High-risk records", format_number(len(high_risk_df))),
    ]

    customer_col = ctx["customer_col"]
    if customer_col and customer_col in high_risk_df.columns:
        lines.append(make_kv_line("Unique high-risk customers", format_number(high_risk_df[customer_col].nunique(dropna=True))))

    score_series = high_risk_df["_risk_numeric"].dropna()
    if not score_series.empty:
        lines.append(make_kv_line("Average high-risk score", f"{score_series.mean():.4f}"))
        lines.append(make_kv_line("Max high-risk score", f"{score_series.max():.4f}"))

    segment_block = _format_top_counts("Top High-Risk Segments", _top_counts(high_risk_df, ctx["segment_col"]))
    country_block = _format_top_counts("Top High-Risk Countries", _top_counts(high_risk_df, ctx["country_col"]))
    channel_block = _format_top_counts("Top High-Risk Channels", _top_counts(high_risk_df, ctx["channel_col"]))

    if segment_block:
        lines.append(segment_block)
    if country_block:
        lines.append(country_block)
    if channel_block:
        lines.append(channel_block)

    lines.append(make_section_title(f"Top {min(top_n, len(high_risk_df))} High-Risk Customers"))
    lines.extend(_format_customer_risk_lines(high_risk_df, ctx, top_n))

    if live_ctx:
        lines.extend(_build_live_risk_lines(live_ctx, limit=min(top_n, 5)))

    return "\n".join(lines)


def get_risk_segment_summary(limit: int = 10) -> str:
    ctx = _prepare_risk_context()

    if ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx["empty"]:
        return make_empty_message("Risk score data is available but empty.")

    df = ctx["df"]
    segment_col = ctx["segment_col"]

    if not segment_col or segment_col not in df.columns:
        return make_empty_message("Segment information is not available in the risk dataset.")

    try:
        grouped = (
            df.groupby(segment_col, dropna=False)
            .agg(
                records=("_risk_bucket", "size"),
                avg_score=("_risk_numeric", "mean"),
                max_score=("_risk_numeric", "max"),
            )
            .reset_index()
            .sort_values(["avg_score", "records"], ascending=[False, False])
            .head(limit)
        )
    except Exception:
        return make_empty_message("Segment information is present, but it could not be summarized.")

    lines = [make_section_title("Risk by Segment")]
    for _, row in grouped.iterrows():
        seg = _clean_value(row[segment_col])
        avg_text = f"{row['avg_score']:.4f}" if pd.notna(row["avg_score"]) else "—"
        max_text = f"{row['max_score']:.4f}" if pd.notna(row["max_score"]) else "—"
        lines.append(
            f"- **{seg}** | Records: {format_number(row['records'])} | Avg Score: {avg_text} | Max Score: {max_text}"
        )

    return "\n".join(lines)


def get_risk_country_summary(limit: int = 10) -> str:
    ctx = _prepare_risk_context()

    if ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx["empty"]:
        return make_empty_message("Risk score data is available but empty.")

    df = ctx["df"]
    country_col = ctx["country_col"]

    if not country_col or country_col not in df.columns:
        return make_empty_message("Country information is not available in the risk dataset.")

    try:
        grouped = (
            df.groupby(country_col, dropna=False)
            .agg(
                records=("_risk_bucket", "size"),
                avg_score=("_risk_numeric", "mean"),
                max_score=("_risk_numeric", "max"),
            )
            .reset_index()
            .sort_values(["records", "avg_score"], ascending=[False, False])
            .head(limit)
        )
    except Exception:
        return make_empty_message("Country information is present, but it could not be summarized.")

    lines = [make_section_title("Risk by Country")]
    for _, row in grouped.iterrows():
        country = _clean_value(row[country_col])
        avg_text = f"{row['avg_score']:.4f}" if pd.notna(row["avg_score"]) else "—"
        max_text = f"{row['max_score']:.4f}" if pd.notna(row["max_score"]) else "—"
        lines.append(
            f"- **{country}** | Records: {format_number(row['records'])} | Avg Score: {avg_text} | Max Score: {max_text}"
        )

    return "\n".join(lines)


def get_risk_channel_summary(limit: int = 10) -> str:
    ctx = _prepare_risk_context()

    if ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx["empty"]:
        return make_empty_message("Risk score data is available but empty.")

    df = ctx["df"]
    channel_col = ctx["channel_col"]

    if not channel_col or channel_col not in df.columns:
        return make_empty_message("Channel information is not available in the risk dataset.")

    try:
        grouped = (
            df.groupby(channel_col, dropna=False)
            .agg(
                records=("_risk_bucket", "size"),
                avg_score=("_risk_numeric", "mean"),
                max_score=("_risk_numeric", "max"),
            )
            .reset_index()
            .sort_values(["records", "avg_score"], ascending=[False, False])
            .head(limit)
        )
    except Exception:
        return make_empty_message("Channel information is present, but it could not be summarized.")

    lines = [make_section_title("Risk by Channel")]
    for _, row in grouped.iterrows():
        channel = _clean_value(row[channel_col])
        avg_text = f"{row['avg_score']:.4f}" if pd.notna(row["avg_score"]) else "—"
        max_text = f"{row['max_score']:.4f}" if pd.notna(row["max_score"]) else "—"
        lines.append(
            f"- **{channel}** | Records: {format_number(row['records'])} | Avg Score: {avg_text} | Max Score: {max_text}"
        )

    return "\n".join(lines)


def get_risk_reason_summary(limit: int = 10) -> str:
    ctx = _prepare_risk_context()

    if ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx["empty"]:
        return make_empty_message("Risk score data is available but empty.")

    df = ctx["df"]
    reason_col = ctx["reason_col"]

    if not reason_col or reason_col not in df.columns:
        return make_empty_message("Risk reason information is not available in the dataset.")

    items = _top_counts(df, reason_col, n=limit)
    if not items:
        return make_empty_message("Risk reason information is present, but it could not be summarized.")

    lines = [make_section_title("Top Risk Reasons")]
    for reason, count in items:
        lines.append(f"- **{reason}**: {format_number(count)}")
    return "\n".join(lines)


def get_risk_status_summary(limit: int = 10) -> str:
    ctx = _prepare_risk_context()

    if ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx["empty"]:
        return make_empty_message("Risk score data is available but empty.")

    df = ctx["df"]
    status_col = ctx["status_col"]

    if not status_col or status_col not in df.columns:
        return make_empty_message("Risk status information is not available in the dataset.")

    items = _top_counts(df, status_col, n=limit)
    if not items:
        return make_empty_message("Risk status information is present, but it could not be summarized.")

    lines = [make_section_title("Risk Status Summary")]
    for status, count in items:
        lines.append(f"- **{status}**: {format_number(count)}")
    return "\n".join(lines)


def get_customer_risk_details(customer_id: str) -> str:
    ctx = _prepare_risk_context()

    if ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx["empty"]:
        return make_empty_message("Risk score data is available but empty.")

    df = ctx["df"]
    customer_col = ctx["customer_col"]

    if not customer_col or customer_col not in df.columns:
        return make_empty_message("Customer information is not available in the risk dataset.")

    matches = df[df[customer_col].astype(str).str.upper() == customer_id.upper()].copy()

    if matches.empty:
        return make_empty_message(f"No risk records were found for customer {customer_id}.")

    try:
        matches = matches.sort_values(
            by=["_risk_numeric", "_amount_numeric"],
            ascending=[False, False],
            na_position="last",
        )
    except Exception:
        pass

    lines = [make_section_title(f"Risk Details for {customer_id}")]
    lines.extend(_format_customer_risk_lines(matches, ctx, limit=min(10, len(matches))))
    return "\n".join(lines)


def get_latest_risk_timestamp() -> str:
    ctx = _prepare_risk_context()
    live_ctx = _load_live_stream_context()

    if ctx is None and live_ctx is None:
        return make_empty_message("Risk score data is unavailable.")

    if ctx is not None and not ctx.get("empty"):
        timestamp_col = ctx["timestamp_col"]
        df = ctx["df"]

        if timestamp_col and timestamp_col in df.columns:
            ts = df["_parsed_timestamp"].dropna()
            if not ts.empty:
                lines = [make_kv_line("Latest batch/cloud risk timestamp", str(ts.max()))]
                if live_ctx:
                    try:
                        live_ts_col = live_ctx.get("timestamp_col")
                        live_df = live_ctx["df"]
                        if live_ts_col and live_ts_col in live_df.columns:
                            live_ts = pd.to_datetime(live_df[live_ts_col], errors="coerce").dropna()
                            if not live_ts.empty:
                                lines.append(make_kv_line("Latest live stream/cloud timestamp", str(live_ts.max())))
                    except Exception:
                        pass
                return "\n".join(lines)

    if live_ctx:
        live_ts_col = live_ctx.get("timestamp_col")
        live_df = live_ctx["df"]
        if live_ts_col and live_ts_col in live_df.columns:
            live_ts = pd.to_datetime(live_df[live_ts_col], errors="coerce").dropna()
            if not live_ts.empty:
                return make_kv_line("Latest live stream/cloud risk timestamp", str(live_ts.max()))

    return make_empty_message("Timestamp information is not available in the risk dataset.")


# =========================================================
# PUBLIC Q&A
# =========================================================
def answer_risk_question(user_query: str) -> str:
    q = user_query.lower().strip()
    top_n = _extract_limit_from_query(q, default=TOP_N_DEFAULT, max_limit=50)

    if not q:
        return ""

    if _contains_any(q, [
        "risk summary",
        "risk overview",
        "current risk overview",
        "show risk summary",
        "show risk overview",
        "overall risk",
        "fraud risk summary",
    ]):
        return get_risk_summary(top_n=top_n)

    if _contains_any(q, [
        "live risk",
        "latest live risk",
        "stream risk",
        "cloud risk",
        "recent risk from cloud",
        "recent live risk",
    ]):
        live_ctx = _load_live_stream_context()
        if not live_ctx:
            return make_empty_message("Live risk data is unavailable.")
        return "\n".join([make_section_title("Live Risk View")] + _build_live_risk_lines(live_ctx, limit=top_n))

    customer_id = _extract_customer_id(user_query)
    if customer_id and _contains_any(q, ["customer", "risk", "score", "details", "show"]):
        return get_customer_risk_details(customer_id)

    if _contains_any(q, [
        "top risky customers",
        "top risk customers",
        "highest risk customers",
        "show top risk",
        "show top risky customers",
        "riskiest customer",
        "riskiest customers",
    ]):
        return get_top_risk_customers(top_n=top_n)

    if _contains_any(q, [
        "high risk",
        "high-risk",
        "high risk summary",
        "high-risk summary",
        "high risk customers",
    ]):
        return get_high_risk_summary(top_n=top_n)

    if _contains_any(q, [
        "risk segment summary",
        "risk by segment",
        "segment risk",
        "customer segment risk",
    ]):
        return get_risk_segment_summary(limit=top_n)

    if _contains_any(q, [
        "risk by country",
        "risk country summary",
        "country risk",
        "high risk countries",
    ]):
        return get_risk_country_summary(limit=top_n)

    if _contains_any(q, [
        "risk by channel",
        "risk channel summary",
        "channel risk",
        "high risk channels",
    ]):
        return get_risk_channel_summary(limit=top_n)

    if _contains_any(q, [
        "risk reason",
        "risk reasons",
        "top risk reasons",
        "why customers are risky",
    ]):
        return get_risk_reason_summary(limit=top_n)

    if _contains_any(q, [
        "risk status",
        "risk statuses",
        "status summary",
        "risk status summary",
    ]):
        return get_risk_status_summary(limit=top_n)

    if _contains_any(q, [
        "latest risk timestamp",
        "latest timestamp",
        "risk timestamp",
        "most recent risk record",
    ]):
        return get_latest_risk_timestamp()

    if _contains_any(q, [
        "average risk score",
        "max risk score",
        "how many high risk",
        "how many medium risk",
        "how many low risk",
    ]):
        return get_risk_summary(top_n=top_n)

    return ""
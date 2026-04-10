from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd

from agentic_ai.config import DATA_PATHS, TOP_N_DEFAULT
from agentic_ai.utils.data_access import safe_load_csv, safe_load_parquet, standardize
from agentic_ai.utils.formatter import format_number
from agentic_ai.utils.presentation_formatter import (
    make_section_title,
    make_kv_line,
    make_bullet_list,
    make_empty_message,
)


# =========================================================
# FILE LOADING
# =========================================================
def _safe_read(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None

    try:
        p = Path(path)
    except Exception:
        return None

    if not p.exists() or not p.is_file():
        return None

    try:
        suffix = p.suffix.lower()
        if suffix == ".csv":
            df = safe_load_csv(p)
        elif suffix == ".parquet":
            df = safe_load_parquet(p)
        else:
            return None

        if df is None or df.empty:
            return None

        return standardize(df)
    except Exception:
        return None


def _candidate_anomaly_paths() -> list[Path]:
    candidates = []

    configured_keys = [
        "anomalies",
        "customer_anomalies",
        "transaction_anomalies",
        "anomaly_summary_parquet",
        "anomaly_summary_csv",
        "risk_anomaly_summary_parquet",
        "risk_anomaly_summary_csv",
        "top_unusual_customers_csv",
        "top_high_risk_customers_csv",
        "cloud_customer_anomalies",
        "cloud_transaction_anomalies",
        "latest_anomaly_parquet",
        "latest_anomaly_csv",
    ]

    for key in configured_keys:
        value = DATA_PATHS.get(key)
        if value:
            candidates.append(Path(value))

    common_paths = [
        Path("outputs/customer_anomalies.parquet"),
        Path("outputs/transaction_anomalies.parquet"),
        Path("outputs/anomaly_summary.parquet"),
        Path("outputs/anomaly_summary.csv"),
        Path("outputs/top_unusual_customers.csv"),
        Path("outputs/top_high_risk_customers.csv"),
        Path("bi_exports/customer_anomalies.parquet"),
        Path("bi_exports/transaction_anomalies.parquet"),
        Path("bi_exports/anomaly_summary.parquet"),
        Path("bi_exports/anomaly_summary.csv"),
        Path("bi_exports/risk_anomaly_summary.parquet"),
        Path("bi_exports/risk_anomaly_summary.csv"),
        Path("data/latest/customer_anomalies.parquet"),
        Path("data/latest/transaction_anomalies.parquet"),
    ]
    candidates.extend(common_paths)

    normalized = []
    seen = set()
    for p in candidates:
        try:
            key = str(p.resolve()) if p.exists() else str(p)
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            normalized.append(p)

    return normalized


# =========================================================
# COLUMN DETECTION
# =========================================================
def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    if df is None or df.empty:
        return None

    lowered = {str(c).strip().lower(): c for c in df.columns}

    for name in candidates:
        hit = lowered.get(name.strip().lower())
        if hit is not None:
            return hit

    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in candidates:
            if cand.strip().lower() in lc:
                return c

    return None


def _customer_col(df): return _find_col(df, "customer_id", "cust_id", "customer", "client_id")
def _txn_col(df): return _find_col(df, "transaction_id", "txn_id", "txn", "id")
def _amount_col(df): return _find_col(df, "amount", "transaction_amount", "txn_amount", "amt", "linked_amount", "total_amount", "spend")
def _channel_col(df): return _find_col(df, "channel", "transaction_channel", "txn_channel")
def _country_col(df): return _find_col(df, "country", "merchant_country", "txn_country", "home_country")
def _segment_col(df): return _find_col(df, "customer_segment", "segment", "label", "cluster")
def _reason_col(df): return _find_col(df, "reason", "anomaly_reason", "risk_reason", "explanation", "driver", "top_driver")
def _timestamp_col(df): return _find_col(df, "timestamp", "txn_timestamp", "event_time", "transaction_time", "datetime", "created_at", "date")
def _score_col(df): return _find_col(df, "anomaly_score", "score", "outlier_score", "prediction_score", "fraud_risk_score", "risk_score", "probability")
def _flag_col(df): return _find_col(df, "is_anomaly", "anomaly", "predicted_anomaly", "outlier", "prediction", "anomaly_flag", "fraud_flag", "is_outlier")


# =========================================================
# HELPERS
# =========================================================
def _safe_numeric_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _safe_flag_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(dtype="float64")

    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "1": 1,
            "true": 1,
            "yes": 1,
            "y": 1,
            "anomaly": 1,
            "outlier": 1,
            "-1": 1,
            "0": 0,
            "false": 0,
            "no": 0,
            "n": 0,
            "normal": 0,
            "inlier": 0,
        })
    )
    return pd.to_numeric(mapped, errors="coerce")


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


def _fmt_score(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.4f}"
    except Exception:
        return "—"


def _fmt_amount(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return format_number(float(value))
    except Exception:
        return "—"


def _extract_limit_from_query(q: str, default: int = 5, max_limit: int = 25) -> int:
    match = re.search(r"\b(top|last|latest|recent|show)\s+(\d+)\b", q)
    if match:
        try:
            value = int(match.group(2))
            return max(1, min(value, max_limit))
        except Exception:
            return default
    return default


def _add_if_present(parts: list[str], label: str, value) -> None:
    cleaned = _clean_value(value)
    if cleaned != "—":
        parts.append(f"{label}: {cleaned}")


# =========================================================
# SOURCE SELECTION
# =========================================================
def _quality_score(df: pd.DataFrame, path: Path) -> int:
    score = 0
    name = str(path).lower()

    if "customer_anomalies.parquet" in name:
        score += 1000
    elif "transaction_anomal" in name:
        score += 950
    elif "anomaly_summary" in name:
        score += 700
    elif "top_unusual_customers" in name:
        score += 500
    elif "top_high_risk_customers" in name:
        score += 400
    elif "risk_anomaly_summary" in name:
        score += 300

    if "bi_exports" in name:
        score += 50
    if "latest" in name:
        score += 60
    if "cloud" in name:
        score += 60

    if "customer_risk_scores_enriched" in name:
        score -= 10000
    if "customer_risk_scores" in name and "anomal" not in name:
        score -= 5000

    if _customer_col(df): score += 20
    if _txn_col(df): score += 35
    if _amount_col(df): score += 25
    if _channel_col(df): score += 20
    if _country_col(df): score += 20
    if _segment_col(df): score += 10
    if _reason_col(df): score += 15
    if _timestamp_col(df): score += 20
    if _score_col(df): score += 25
    if _flag_col(df): score += 15

    detail_count = sum([
        _txn_col(df) is not None,
        _amount_col(df) is not None,
        _channel_col(df) is not None,
        _country_col(df) is not None,
        _timestamp_col(df) is not None,
        _reason_col(df) is not None,
        _score_col(df) is not None or _flag_col(df) is not None,
    ])

    if detail_count <= 2:
        score -= 150
    elif detail_count >= 5:
        score += 80

    try:
        score += int(path.stat().st_mtime // 100000)
    except Exception:
        pass

    return score


def _load_best_anomaly_source() -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    best_df = None
    best_path = None
    best_score = -10**9

    for path in _candidate_anomaly_paths():
        df = _safe_read(path)
        if df is None or df.empty:
            continue

        q = _quality_score(df, path)
        if q > best_score:
            best_score = q
            best_df = df
            best_path = path

    return best_df, best_path


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


def _score_streaming_suspicion(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
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

    working["_suspicion_score"] = 0
    working["_suspicion_reasons"] = ""

    def add_reason(mask, points, reason_text):
        if mask is None:
            return
        try:
            mask = mask.fillna(False)
        except Exception:
            return

        working.loc[mask, "_suspicion_score"] += points
        working.loc[mask, "_suspicion_reasons"] = (
            working.loc[mask, "_suspicion_reasons"]
            .astype(str)
            .apply(lambda x: f"{x}; {reason_text}".strip("; ").strip())
        )

    amt = pd.to_numeric(working.get("_amount_numeric"), errors="coerce")

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
        add_reason(channel_lower.eq("atm"), 1, "atm cash movement")
        add_reason(channel_lower.eq("ecom"), 1, "ecommerce channel")
        add_reason(channel_lower.eq("mobile"), 1, "mobile channel")

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
        .replace("", "no major real-time anomaly rules were triggered")
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


# =========================================================
# PRESENTATION HELPERS
# =========================================================
def _build_record_block(row, txn_col, customer_col, amount_col, channel_col, country_col, segment_col, score_value, reason_value, timestamp_col) -> str:
    title = _clean_value(row.get(customer_col)) if customer_col else "—"
    txn = _clean_value(row.get(txn_col)) if txn_col else "—"

    parts = []
    if txn != "—" and txn != title:
        parts.append(f"Txn: {txn}")

    if amount_col:
        amt = _fmt_amount(row.get(amount_col))
        if amt != "—":
            parts.append(f"Amount: {amt}")

    if score_value not in [None, "—"]:
        parts.append(f"Score: {score_value}")

    if segment_col:
        _add_if_present(parts, "Segment", row.get(segment_col))

    if channel_col:
        _add_if_present(parts, "Channel", row.get(channel_col))

    if country_col:
        _add_if_present(parts, "Country", row.get(country_col))

    if timestamp_col:
        _add_if_present(parts, "Timestamp", row.get(timestamp_col))

    if reason_value and reason_value != "—":
        parts.append(f"Reason: {reason_value}")

    return f"- **{title}**\n  " + " | ".join(parts)


def _top_records_from_df(df: pd.DataFrame, top_n: int = TOP_N_DEFAULT) -> list[str]:
    if df is None or df.empty:
        return []

    customer_col = _customer_col(df)
    txn_col = _txn_col(df)
    amount_col = _amount_col(df)
    channel_col = _channel_col(df)
    country_col = _country_col(df)
    segment_col = _segment_col(df)
    reason_col = _reason_col(df)
    timestamp_col = _timestamp_col(df)
    score_col = _score_col(df)
    flag_col = _flag_col(df)

    working_df = _sort_best_available(df).head(top_n)

    lines = []
    for _, row in working_df.iterrows():
        score_text = _fmt_score(row[score_col]) if score_col else "—"
        if score_text == "—" and flag_col:
            score_text = _clean_value(row[flag_col])

        reason_text = _clean_value(row[reason_col]) if reason_col else "—"

        lines.append(
            _build_record_block(
                row=row,
                txn_col=txn_col,
                customer_col=customer_col,
                amount_col=amount_col,
                channel_col=channel_col,
                country_col=country_col,
                segment_col=segment_col,
                score_value=score_text,
                reason_value=reason_text,
                timestamp_col=timestamp_col,
            )
        )

    return lines


def _sort_best_available(df: pd.DataFrame) -> pd.DataFrame:
    score_col = _score_col(df)
    if score_col:
        scores = _safe_numeric_series(df, score_col)
        return (
            df.assign(__sort_score=scores)
            .sort_values("__sort_score", ascending=False, na_position="last")
            .drop(columns="__sort_score")
        )

    amount_col = _amount_col(df)
    if amount_col:
        amounts = _safe_numeric_series(df, amount_col)
        return (
            df.assign(__sort_amt=amounts)
            .sort_values("__sort_amt", ascending=False, na_position="last")
            .drop(columns="__sort_amt")
        )

    return df


def _generate_key_insight(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No anomaly insight is available."

    segment_col = _segment_col(df)
    amount_col = _amount_col(df)
    score_col = _score_col(df)

    insights = []

    if segment_col:
        top_segment = df[segment_col].astype(str).value_counts().head(1)
        if not top_segment.empty:
            insights.append(f"Most anomaly activity is concentrated in **segment {top_segment.index[0]}**.")

    if amount_col:
        amt = _safe_numeric_series(df, amount_col).dropna()
        if not amt.empty and amt.mean() > 0:
            insights.append(f"Flagged records carry a high financial exposure, with an average suspicious amount of **{format_number(amt.mean())}**.")

    if score_col:
        score = _safe_numeric_series(df, score_col).dropna()
        if not score.empty:
            high_count = int((score >= 0.85).sum())
            if high_count > 0:
                insights.append(f"There are **{format_number(high_count)} high-severity anomaly records** that deserve priority review.")

    return insights[0] if insights else "Suspicious activity is present, but additional detail columns would improve explainability."


def _build_live_anomaly_lines(ctx: dict, limit: int = 5) -> list[str]:
    df = ctx["df"]
    source_path = ctx.get("source_path")
    scored = _score_streaming_suspicion(df, ctx)

    txn_col = ctx.get("txn_col")
    customer_col = ctx.get("customer_col")
    amount_col = ctx.get("amount_col")
    channel_col = ctx.get("channel_col")
    country_col = ctx.get("country_col")
    customer_segment_col = ctx.get("customer_segment_col")
    timestamp_col = ctx.get("timestamp_col")

    high_df = scored[scored["_suspicion_bucket"] == "High"]
    med_df = scored[scored["_suspicion_bucket"] == "Medium"]
    low_df = scored[scored["_suspicion_bucket"] == "Low"]

    lines = [make_section_title("Live Monitoring Snapshot")]

    if source_path is not None:
        try:
            lines.append(make_kv_line("Latest batch", source_path.name))
        except Exception:
            lines.append(make_kv_line("Latest batch", str(source_path)))

    lines.append(make_kv_line("Records evaluated", format_number(len(scored))))
    lines.append(make_kv_line("High Suspicion", format_number(len(high_df))))
    lines.append(make_kv_line("Medium Suspicion", format_number(len(med_df))))
    lines.append(make_kv_line("Low Suspicion", format_number(len(low_df))))

    try:
        lines.append(make_kv_line("Average live suspicion score", f"{scored['_suspicion_score'].mean():.2f}"))
        lines.append(make_kv_line("Max live suspicion score", f"{scored['_suspicion_score'].max():.2f}"))
    except Exception:
        pass

    lines.append(make_section_title("Top Live Suspicious Records"))
    for _, row in scored.head(limit).iterrows():
        lines.append(
            _build_record_block(
                row=row,
                txn_col=txn_col,
                customer_col=customer_col,
                amount_col=amount_col,
                channel_col=channel_col,
                country_col=country_col,
                segment_col=customer_segment_col,
                score_value=_clean_value(row.get("_suspicion_score")),
                reason_value=_clean_value(row.get("_suspicion_reasons")),
                timestamp_col=timestamp_col,
            )
        )

    return lines


# =========================================================
# PUBLIC SUMMARY
# =========================================================
def get_anomaly_summary(top_n: int = TOP_N_DEFAULT) -> str:
    df, source_path = _load_best_anomaly_source()
    live_ctx = _load_live_stream_context()

    if df is None and live_ctx is None:
        return make_empty_message("Anomaly data is unavailable.")

    if df is None:
        lines = [
            make_section_title("Anomaly Summary"),
            make_empty_message("Batch anomaly data is unavailable."),
        ]
        lines.extend(_build_live_anomaly_lines(live_ctx, limit=top_n))
        return "\n".join(lines)

    customer_col = _customer_col(df)
    txn_col = _txn_col(df)
    amount_col = _amount_col(df)
    score_col = _score_col(df)
    flag_col = _flag_col(df)

    lines = [
        make_section_title("Anomaly Summary"),
        make_section_title("Historical Summary"),
    ]

    if source_path:
        lines.append(make_kv_line("Source file", str(source_path)))

    lines.append(make_kv_line("Records evaluated", format_number(len(df))))

    if customer_col:
        lines.append(make_kv_line("Unique customers", format_number(df[customer_col].nunique())))

    if txn_col:
        lines.append(make_kv_line("Unique transactions", format_number(df[txn_col].nunique())))

    score_series = _safe_numeric_series(df, score_col).dropna()
    if not score_series.empty:
        high_count = int((score_series >= 0.85).sum())
        med_count = int(((score_series >= 0.60) & (score_series < 0.85)).sum())
        low_count = int((score_series < 0.60).sum())

        lines.append(make_kv_line("Average anomaly score", _fmt_score(score_series.mean())))
        lines.append(make_kv_line("Max anomaly score", _fmt_score(score_series.max())))
        lines.append(
            make_bullet_list(
                "Suspicion Breakdown",
                [
                    f"High Suspicion: {format_number(high_count)}",
                    f"Medium Suspicion: {format_number(med_count)}",
                    f"Low Suspicion: {format_number(low_count)}",
                ],
            )
        )
    else:
        flag_series = _safe_flag_series(df, flag_col).dropna()
        if not flag_series.empty:
            lines.append(
                make_bullet_list(
                    "Flag Summary",
                    [
                        f"Flagged anomaly records: {format_number(int((flag_series == 1).sum()))}",
                        f"Non-anomaly records: {format_number(int((flag_series == 0).sum()))}",
                    ],
                )
            )

    if amount_col:
        amount_series = _safe_numeric_series(df, amount_col).dropna()
        if not amount_series.empty:
            lines.append(
                make_bullet_list(
                    "Financial Exposure",
                    [
                        f"Total suspicious amount: {format_number(amount_series.sum())}",
                        f"Average suspicious amount: {format_number(amount_series.mean())}",
                        f"Max suspicious amount: {format_number(amount_series.max())}",
                    ],
                )
            )

    lines.append(make_section_title("Top Suspicious Records"))
    lines.extend(_top_records_from_df(df, top_n=top_n) or ["No suspicious records available."])

    if live_ctx:
        lines.extend(_build_live_anomaly_lines(live_ctx, limit=top_n))

    lines.append(make_section_title("Key Insight"))
    lines.append(_generate_key_insight(df))

    return "\n".join(lines)


# =========================================================
# PUBLIC Q&A
# =========================================================
def answer_anomaly_question(user_query: str, top_n: int = TOP_N_DEFAULT) -> str:
    q = (user_query or "").strip().lower()
    limit = _extract_limit_from_query(q, default=top_n, max_limit=25)

    if not q:
        return ""

    if any(word in q for word in [
        "live anomaly",
        "latest anomaly from stream",
        "latest anomaly from cloud",
        "recent suspicious streaming",
        "recent suspicious cloud",
        "stream anomaly",
        "cloud anomaly",
        "live suspicious transactions",
    ]):
        live_ctx = _load_live_stream_context()
        if not live_ctx:
            return make_empty_message("Live anomaly data is unavailable.")
        return "\n".join([make_section_title("Live Anomaly View")] + _build_live_anomaly_lines(live_ctx, limit=limit))

    df, source_path = _load_best_anomaly_source()
    live_ctx = _load_live_stream_context()

    if df is None and live_ctx is None:
        return make_empty_message("Anomaly data is unavailable.")

    if any(word in q for word in ["summary", "overview", "show anomalies", "anomaly summary"]):
        return get_anomaly_summary(top_n=limit)

    if live_ctx and any(word in q for word in ["latest", "recent", "live", "stream", "cloud"]):
        if any(word in q for word in ["suspicious", "anomaly", "flagged"]):
            return "\n".join([make_section_title("Live Anomaly View")] + _build_live_anomaly_lines(live_ctx, limit=limit))

    if df is None:
        return "\n".join([
            make_section_title("Anomaly Summary"),
            make_empty_message("Batch anomaly data is unavailable."),
        ] + _build_live_anomaly_lines(live_ctx, limit=limit))

    score_col = _score_col(df)
    flag_col = _flag_col(df)
    customer_col = _customer_col(df)
    channel_col = _channel_col(df)
    country_col = _country_col(df)
    timestamp_col = _timestamp_col(df)
    amount_col = _amount_col(df)
    reason_col = _reason_col(df)
    segment_col = _segment_col(df)
    txn_col = _txn_col(df)

    if "high risk" in q or "top suspicious" in q:
        if score_col:
            scores = _safe_numeric_series(df, score_col)
            filtered = df.loc[scores >= 0.85].copy()
            if filtered.empty:
                filtered = _sort_best_available(df).head(limit)
        elif flag_col:
            flags = _safe_flag_series(df, flag_col)
            filtered = df.loc[flags == 1].copy()
            if filtered.empty:
                filtered = _sort_best_available(df).head(limit)
        else:
            filtered = _sort_best_available(df).head(limit)

        lines = [
            make_section_title("Top Suspicious Records"),
            make_kv_line("Source", str(source_path)),
        ]
        lines.extend(_top_records_from_df(filtered, top_n=limit))
        return "\n".join(lines)

    if customer_col and "customer" in q:
        for token in user_query.replace(",", " ").split():
            tok = token.strip().upper()
            if tok.startswith("C"):
                matches = df[df[customer_col].astype(str).str.upper() == tok]
                if not matches.empty:
                    lines = [make_section_title(f"Anomaly Records for {tok}")]
                    lines.extend(_top_records_from_df(matches, top_n=limit))
                    return "\n".join(lines)

    if channel_col:
        for ch in ["ATM", "POS", "ECOM", "MOBILE", "BRANCH"]:
            if ch.lower() in q:
                matches = df[df[channel_col].astype(str).str.upper() == ch]
                if matches.empty:
                    return make_empty_message(f"No suspicious {ch} records were found.")
                lines = [make_section_title(f"Suspicious {ch} Records")]
                lines.extend(_top_records_from_df(matches, top_n=limit))
                return "\n".join(lines)

    if country_col and "country" in q:
        counts = df[country_col].astype(str).value_counts().head(10)
        lines = [make_section_title("Top Countries in Anomaly Records")]
        for country, count in counts.items():
            lines.append(f"- **{country}**: {format_number(count)}")
        return "\n".join(lines)

    if segment_col and "segment" in q:
        counts = df[segment_col].astype(str).value_counts().head(10)
        lines = [make_section_title("Top Customer Segments in Anomaly Records")]
        for segment, count in counts.items():
            lines.append(f"- **{segment}**: {format_number(count)}")
        return "\n".join(lines)

    if timestamp_col and any(word in q for word in ["timestamp", "earliest", "oldest", "latest"]):
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        if ts.notna().sum() == 0:
            return make_empty_message("Timestamp data is unavailable in the anomaly source.")
        if "earliest" in q or "oldest" in q:
            return make_kv_line("Earliest anomaly timestamp", str(ts.min()))
        return make_kv_line("Latest anomaly timestamp", str(ts.max()))

    if amount_col and any(word in q for word in ["amount", "largest", "biggest", "max amount"]):
        amt = _safe_numeric_series(df, amount_col)
        if amt.notna().sum() == 0:
            return make_empty_message("Amount data is unavailable in the anomaly source.")

        idx = amt.idxmax()
        row = df.loc[idx]

        score_text = _fmt_score(row[score_col]) if score_col else (_clean_value(row[flag_col]) if flag_col else "—")
        reason_text = _clean_value(row[reason_col]) if reason_col else "—"

        return "\n".join([
            make_section_title("Largest Suspicious Record"),
            _build_record_block(
                row=row,
                txn_col=txn_col,
                customer_col=customer_col,
                amount_col=amount_col,
                channel_col=channel_col,
                country_col=country_col,
                segment_col=segment_col,
                score_value=score_text,
                reason_value=reason_text,
                timestamp_col=timestamp_col,
            )
        ])

    if reason_col and "reason" in q:
        counts = df[reason_col].astype(str).value_counts().head(10)
        lines = [make_section_title("Top Anomaly Reasons")]
        for reason, count in counts.items():
            lines.append(f"- **{reason}**: {format_number(count)}")
        return "\n".join(lines)

    # IMPORTANT:
    # Do not dump full summary for unmatched anomaly questions.
    return ""
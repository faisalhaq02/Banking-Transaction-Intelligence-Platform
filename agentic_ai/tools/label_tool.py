from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from agentic_ai.config import DATA_PATHS, PREFERRED_PATHS, TOP_N_DEFAULT
from agentic_ai.utils.data_access import (
    safe_load_first_available,
    safe_load_latest_file_from_dir,
    standardize,
)
from agentic_ai.utils.formatter import format_number


LABEL_HINTS = [
    "label",
    "segment",
    "category",
    "risk",
    "status",
    "channel",
    "merchant",
    "country",
    "city",
    "band",
]


# -------------------------------------------------
# LOADING
# -------------------------------------------------
def _load_label_df():
    """
    Prefer latest streaming-style transactional data first,
    then fallback to segment summary style data.
    """
    df = safe_load_first_available(PREFERRED_PATHS.get("streaming_transactions", []))

    if df is None:
        df = safe_load_latest_file_from_dir(DATA_PATHS.get("stream_batches_dir"))

    if df is None:
        df = safe_load_first_available(PREFERRED_PATHS.get("segment_summary", []))

    if df is None:
        return None

    df = standardize(df)
    if df.empty:
        return None

    return df


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases)


def _extract_limit_from_query(q: str, default: int = TOP_N_DEFAULT, max_limit: int = 50) -> int:
    match = re.search(r"\b(top|show|last|first)\s+(\d+)\b", q)
    if match:
        try:
            value = int(match.group(2))
            return max(1, min(value, max_limit))
        except Exception:
            return default
    return default


def _extract_customer_id(user_query: str) -> Optional[str]:
    match = re.search(r"\bC\d+\b", user_query.upper())
    if match:
        return match.group(0)
    return None


def _find_label_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = str(c).lower().strip()
        if any(h in lc for h in LABEL_HINTS):
            cols.append(c)
    return cols


def _find_customer_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["customer_id", "cust_id", "customer", "client_id"]:
        if col in df.columns:
            return col
    return None


def _match_requested_label_column(df: pd.DataFrame, q: str) -> Optional[str]:
    """
    Try to infer which label-like column the user is asking about.
    """
    label_cols = _find_label_columns(df)
    if not label_cols:
        return None

    for col in label_cols:
        if col.lower() in q:
            return col

    alias_map = {
        "segment": ["segment", "customer segment", "segments"],
        "label": ["label", "labels"],
        "risk": ["risk", "risk label", "risk bucket", "risk status"],
        "status": ["status", "statuses"],
        "channel": ["channel", "channels"],
        "merchant": ["merchant", "merchant category", "merchant categories", "category", "categories"],
        "country": ["country", "countries"],
        "city": ["city", "cities"],
        "band": ["band", "age band", "customer age band"],
    }

    for canonical, aliases in alias_map.items():
        if any(alias in q for alias in aliases):
            for col in label_cols:
                if canonical in col.lower():
                    return col

    return None


def _top_counts(df: pd.DataFrame, col: str, n: int = 10):
    try:
        counts = df[col].fillna("Unknown").astype(str).value_counts().head(n)
        return list(counts.items())
    except Exception:
        return []


def _format_counts(title: str, items) -> str:
    if not items:
        return f"{title}\nNo values available."

    lines = [title]
    for key, value in items:
        lines.append(f"- {key}: {format_number(value)}")
    return "\n".join(lines)


# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
def get_label_summary() -> str:
    df = _load_label_df()
    if df is None:
        return "Label data is unavailable."

    label_cols = _find_label_columns(df)
    if not label_cols:
        return "No label-like columns were detected in the latest data."

    lines = [
        "Label summary:",
        f"- Rows: {format_number(len(df))}",
        f"- Detected label-like columns: {format_number(len(label_cols))}",
        "- Available label columns:",
    ]

    for col in label_cols[:20]:
        try:
            unique_count = df[col].nunique(dropna=True)
        except Exception:
            unique_count = 0
        lines.append(f"  - {col} (unique values: {format_number(unique_count)})")

    lines.append("")
    lines.append("Top values for key label columns:")

    for col in label_cols[:8]:
        counts = _top_counts(df, col, n=5)
        lines.append(f"- {col}")
        if counts:
            for key, value in counts:
                lines.append(f"  - {key}: {format_number(value)}")
        else:
            lines.append("  - No values available.")

    return "\n".join(lines)


# -------------------------------------------------
# DETAILED ANSWERS
# -------------------------------------------------
def _get_available_labels_answer(df: pd.DataFrame) -> str:
    label_cols = _find_label_columns(df)
    if not label_cols:
        return "No label-like columns were detected in the data."

    lines = ["Available label-like columns:"]
    for col in label_cols:
        sample_vals = list(df[col].dropna().astype(str).unique()[:5])
        sample_text = ", ".join(sample_vals) if sample_vals else "No sample values"
        lines.append(f"- {col}: {sample_text}")
    return "\n".join(lines)


def _get_column_breakdown(df: pd.DataFrame, col: str, limit: int = 10) -> str:
    counts = _top_counts(df, col, n=limit)
    return _format_counts(f"Breakdown for {col}:", counts)


def _get_customer_label_details(df: pd.DataFrame, customer_id: str) -> str:
    customer_col = _find_customer_col(df)
    if not customer_col:
        return "Customer information is not available in the label dataset."

    matches = df[df[customer_col].astype(str).str.upper() == customer_id.upper()].copy()

    if matches.empty:
        return f"No label records were found for customer {customer_id}."

    label_cols = _find_label_columns(matches)
    if not label_cols:
        return f"Records were found for customer {customer_id}, but no label-like columns were detected."

    lines = [f"Label details for customer {customer_id}:"]
    for col in label_cols:
        vals = matches[col].fillna("Unknown").astype(str).value_counts()
        if len(vals) == 1:
            lines.append(f"- {col}: {vals.index[0]}")
        else:
            parts = [f"{k} ({format_number(v)})" for k, v in vals.items()]
            lines.append(f"- {col}: {', '.join(parts)}")

    return "\n".join(lines)


def _get_label_value_search(df: pd.DataFrame, user_query: str, limit: int = 10) -> Optional[str]:
    """
    Search for records matching a label value mentioned in the question.
    Example: 'show premium labels' or 'show high risk label records'
    """
    q = user_query.lower().strip()
    label_cols = _find_label_columns(df)
    if not label_cols:
        return None

    candidate_values = []
    for col in label_cols:
        try:
            vals = df[col].dropna().astype(str).unique()[:200]
            candidate_values.extend([(col, str(v)) for v in vals])
        except Exception:
            continue

    matches = []
    for col, value in candidate_values:
        value_lower = value.lower()
        if value_lower and value_lower in q and len(value_lower) >= 2:
            count = int((df[col].fillna("").astype(str).str.lower() == value_lower).sum())
            if count > 0:
                matches.append((col, value, count))

    if not matches:
        return None

    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    best_col, best_value, best_count = matches[0]

    lines = [
        f"Records for {best_col} = {best_value}:",
        f"- Matching rows: {format_number(best_count)}",
    ]

    preview = (
        df[df[best_col].fillna("").astype(str).str.lower() == best_value.lower()]
        .head(limit)
        .copy()
    )

    customer_col = _find_customer_col(preview)
    if customer_col:
        lines.append("- Sample customers:")
        for val in preview[customer_col].fillna("Unknown").astype(str).head(limit):
            lines.append(f"  - {val}")

    return "\n".join(lines)


# -------------------------------------------------
# PUBLIC QA
# -------------------------------------------------
def answer_label_question(question: str) -> str | None:
    df = _load_label_df()
    if df is None:
        return "Label data is unavailable."

    q = question.lower().strip()
    limit = _extract_limit_from_query(q, default=TOP_N_DEFAULT, max_limit=50)

    if not q:
        return ""

    label_cols = _find_label_columns(df)
    if not label_cols:
        return "No label-like columns were detected in the latest data."

    # Customer-specific lookup
    customer_id = _extract_customer_id(question)
    if customer_id and _contains_any(q, ["customer", "label", "labels", "show", "details", "segment", "risk", "status"]):
        return _get_customer_label_details(df, customer_id)

    # Explicit summary request
    if _contains_any(q, [
        "label summary",
        "summarize labels",
        "labels overview",
        "label overview",
    ]):
        return get_label_summary()

    # What labels exist?
    if _contains_any(q, [
        "new label",
        "new labels",
        "what labels",
        "available labels",
        "which labels",
        "show labels",
        "what label columns",
        "label columns",
    ]):
        return _get_available_labels_answer(df)

    # Requested label column breakdown
    requested_col = _match_requested_label_column(df, q)
    if requested_col:
        if _contains_any(q, ["top", "breakdown", "distribution", "summary", "count", "counts", "show"]):
            return _get_column_breakdown(df, requested_col, limit=limit)

    # Search for a label value inside the query
    value_match_answer = _get_label_value_search(df, question, limit=limit)
    if value_match_answer:
        return value_match_answer

    # Generic segment/risk/status/channel/country/city type questions
    if _contains_any(q, ["segment", "risk", "status", "channel", "merchant", "country", "city", "category", "band"]):
        requested_col = _match_requested_label_column(df, q)
        if requested_col:
            return _get_column_breakdown(df, requested_col, limit=limit)

    # IMPORTANT:
    # Do not dump full summary for unmatched label questions.
    return ""
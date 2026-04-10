from __future__ import annotations

from agentic_ai.config import DATA_PATHS, PREFERRED_PATHS
from agentic_ai.utils.data_access import (
    safe_load_first_available,
    safe_load_latest_file_from_dir,
    standardize,
)
from agentic_ai.utils.formatter import format_number


SEGMENT_HINTS = [
    "customer_segment", "segment", "cluster", "group", "label"
]

CUSTOMER_HINTS = [
    "customer_id", "cust_id", "customer"
]

AMOUNT_HINTS = [
    "amount", "total_amount", "total_spend", "spend", "transaction_amount"
]


def _load_segment_df():
    # Try explicit segment outputs first
    df = safe_load_first_available(PREFERRED_PATHS.get("customer_segments", []))

    # Fallbacks
    if df is None:
        df = safe_load_first_available(PREFERRED_PATHS.get("customer_risk_scores", []))

    if df is None:
        df = safe_load_first_available(PREFERRED_PATHS.get("customer_risk_scores_enriched", []))

    if df is None:
        df = safe_load_latest_file_from_dir(DATA_PATHS.get("outputs_dir"))

    if df is None:
        return None

    df = standardize(df)
    if df.empty:
        return None

    return df


def _find_first(df, candidates):
    for c in df.columns:
        if str(c).strip().lower() in [x.lower() for x in candidates]:
            return c

    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def get_segment_summary(top_n: int = 10) -> str:
    df = _load_segment_df()

    if df is None:
        return "Customer segment data is unavailable."

    seg_col = _find_first(df, SEGMENT_HINTS)
    cust_col = _find_first(df, CUSTOMER_HINTS)
    amt_col = _find_first(df, AMOUNT_HINTS)

    if seg_col is None:
        return f"Segment file exists, but no segment column was found. Available columns: {list(df.columns)}"

    lines = []

    lines.append("Customer segment summary:")
    lines.append(f"- Rows: {format_number(len(df))}")

    if cust_col is not None:
        lines.append(f"- Unique customers: {format_number(df[cust_col].nunique())}")

    segment_counts = df[seg_col].astype(str).value_counts(dropna=False)

    lines.append(f"- Unique segments: {format_number(segment_counts.shape[0])}")
    lines.append("- Segment distribution:")

    for seg, cnt in segment_counts.head(top_n).items():
        pct = (cnt / len(df)) * 100 if len(df) else 0
        lines.append(f"  - {seg}: {format_number(cnt)} ({pct:.2f}%)")

    if amt_col is not None:
        try:
            grouped = df.groupby(seg_col)[amt_col].agg(["sum", "mean", "max"]).sort_values("sum", ascending=False)
            lines.append("- Top segments by linked amount:")
            for seg, row in grouped.head(5).iterrows():
                lines.append(
                    f"  - {seg}: total={format_number(row['sum'])}, "
                    f"avg={format_number(row['mean'])}, max={format_number(row['max'])}"
                )
        except Exception:
            pass

    return "\n".join(lines)


def answer_segment_question(user_query: str) -> str:
    df = _load_segment_df()

    if df is None:
        return "Customer segment data is unavailable."

    q = user_query.strip().lower()

    seg_col = _find_first(df, SEGMENT_HINTS)
    cust_col = _find_first(df, CUSTOMER_HINTS)
    amt_col = _find_first(df, AMOUNT_HINTS)

    if seg_col is None:
        return "Segment data exists, but no segment column was found."

    segment_counts = df[seg_col].astype(str).value_counts(dropna=False)

    if "largest segment" in q or "biggest segment" in q:
        seg = segment_counts.idxmax()
        cnt = segment_counts.max()
        return f"The largest customer segment is '{seg}' with {format_number(cnt)} customers."

    if "smallest segment" in q:
        seg = segment_counts.idxmin()
        cnt = segment_counts.min()
        return f"The smallest customer segment is '{seg}' with {format_number(cnt)} customers."

    if "segment breakdown" in q or "customer segments" in q or "summarize customer segments" in q or "segment summary" in q:
        return get_segment_summary()

    if ("high value" in q or "mass market" in q or "affluent" in q) and seg_col is not None:
        matches = df[df[seg_col].astype(str).str.lower().str.contains(q.split()[0], na=False)]
        if not matches.empty:
            return f"Found {format_number(len(matches))} records matching that segment."
        return "That segment name was not found in the current dataset."

    return get_segment_summary()
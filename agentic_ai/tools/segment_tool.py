from __future__ import annotations

from typing import Optional

import pandas as pd

from agentic_ai.config import DATA_PATHS
from agentic_ai.utils.data_access import safe_load_csv, safe_load_parquet, standardize
from agentic_ai.utils.formatter import format_number
from agentic_ai.utils.presentation_formatter import (
    clean_value,
    section_title,
    stat_line,
    bullet_line,
    subheading,
    spacer,
)


def _safe_read(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None

    path_str = str(path).strip()
    if not path_str:
        return None

    lower = path_str.lower()

    try:
        if lower.endswith(".csv"):
            df = safe_load_csv(path_str)
        elif lower.endswith(".parquet"):
            df = safe_load_parquet(path_str)
        else:
            return None

        if df is None or df.empty:
            return None

        return standardize(df)
    except Exception:
        return None


def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    if df is None or df.empty:
        return None

    cols = {str(c).strip().lower(): c for c in df.columns}

    for name in candidates:
        hit = cols.get(str(name).strip().lower())
        if hit is not None:
            return hit

    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in candidates:
            if cand.strip().lower() in lc:
                return c

    return None


def _customer_col(df): return _find_col(df, "customer_id", "cust_id", "customer")
def _segment_col(df): return _find_col(df, "customer_segment", "segment", "label", "cluster")
def _amount_col(df): return _find_col(df, "amount", "total_amount", "total_spend", "spend", "linked_amount")
def _risk_col(df): return _find_col(df, "risk_score", "score", "fraud_risk_score", "anomaly_score")


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


def _load_best_segment_source() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    candidates = [
        DATA_PATHS.get("customer_segments"),
        "outputs/customer_segments.parquet",
        "outputs/customer_segments.csv",
        "bi_exports/customer_segments.parquet",
        "bi_exports/customer_segments.csv",
        "outputs/customer_risk_scores_enriched.csv",
        "outputs/customer_risk_scores.parquet",
    ]

    best_df = None
    best_path = None
    best_score = -1
    seen = set()

    for path in candidates:
        if not path:
            continue

        path_str = str(path).strip()
        if not path_str or path_str in seen:
            continue
        seen.add(path_str)

        df = _safe_read(path_str)
        if df is None or df.empty:
            continue

        score = 0
        if "customer_segments" in path_str.lower():
            score += 100
        if _segment_col(df):
            score += 50
        if _customer_col(df):
            score += 20
        if _amount_col(df):
            score += 10
        if _risk_col(df):
            score += 10

        if score > best_score:
            best_score = score
            best_df = df
            best_path = path_str

    return best_df, best_path


def _build_segment_distribution(df: pd.DataFrame, segment_col: str, top_n: int = 10) -> list[str]:
    counts = df[segment_col].fillna("Unknown").astype(str).value_counts(dropna=False)

    lines = [subheading("Segment Distribution")]
    for seg, cnt in counts.head(top_n).items():
        pct = (cnt / len(df)) * 100 if len(df) else 0
        lines.append(bullet_line(clean_value(seg), f"{format_number(cnt)} ({pct:.2f}%)"))
    return lines


def _build_segment_amount_summary(df: pd.DataFrame, segment_col: str, amount_col: str, top_n: int = 5) -> list[str]:
    amt = pd.to_numeric(df[amount_col], errors="coerce")
    grouped = (
        df.assign(__amt=amt)
        .groupby(segment_col, dropna=False)["__amt"]
        .agg(["sum", "mean", "max"])
        .sort_values("sum", ascending=False)
        .head(top_n)
    )

    if grouped.empty:
        return []

    lines = [subheading("Top Segments by Amount")]
    for seg, row in grouped.iterrows():
        lines.append(
            bullet_line(
                clean_value(seg),
                f"Total: {_format_amount(row['sum'])} | "
                f"Average: {_format_amount(row['mean'])} | "
                f"Max: {_format_amount(row['max'])}"
            )
        )
    return lines


def _build_segment_risk_summary(df: pd.DataFrame, segment_col: str, risk_col: str, top_n: int = 5) -> list[str]:
    risk = pd.to_numeric(df[risk_col], errors="coerce")
    grouped = (
        df.assign(__risk=risk)
        .groupby(segment_col, dropna=False)["__risk"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    if grouped.empty:
        return []

    lines = [subheading("Average Risk Score by Segment")]
    for seg, val in grouped.items():
        lines.append(bullet_line(clean_value(seg), _format_score(val)))
    return lines


def _generate_key_insight(df: pd.DataFrame, segment_col: str, amount_col: Optional[str], risk_col: Optional[str]) -> str:
    counts = df[segment_col].fillna("Unknown").astype(str).value_counts(dropna=False)

    if not counts.empty:
        top_seg = counts.index[0]
        top_cnt = counts.iloc[0]
        return f"The largest customer group is {clean_value(top_seg)}, with {format_number(top_cnt)} records."

    if amount_col:
        amt = pd.to_numeric(df[amount_col], errors="coerce").dropna()
        if not amt.empty:
            return f"Segmented customers show an average linked amount of {format_number(amt.mean())}."

    if risk_col:
        risk = pd.to_numeric(df[risk_col], errors="coerce").dropna()
        if not risk.empty:
            return f"Segment-level risk information is available, with an overall average score of {risk.mean():.4f}."

    return "Segment data is available, but more descriptive labels would improve interpretability."


def get_segment_summary(top_n: int = 10) -> str:
    df, source_path = _load_best_segment_source()

    if df is None or source_path is None:
        return "Customer segment data is unavailable."

    if df.empty:
        return "Customer segment data is available but empty."

    segment_col = _segment_col(df)
    customer_col = _customer_col(df)
    amount_col = _amount_col(df)
    risk_col = _risk_col(df)

    if segment_col is None:
        return f"Segment data exists, but no segment column was found. Available columns: {list(df.columns)}"

    segment_counts = df[segment_col].fillna("Unknown").astype(str).value_counts(dropna=False)

    lines = [
        section_title("Customer Segment Summary"),
        spacer(),
        subheading("Overview"),
        stat_line("Source file", source_path),
        stat_line("Rows", format_number(len(df))),
        stat_line("Unique segments", format_number(len(segment_counts))),
    ]

    if customer_col:
        try:
            lines.append(stat_line("Unique customers", format_number(df[customer_col].nunique(dropna=True))))
        except Exception:
            pass

    lines.append(spacer())
    lines.extend(_build_segment_distribution(df, segment_col, top_n=top_n))

    if amount_col:
        amount_lines = _build_segment_amount_summary(df, segment_col, amount_col, top_n=5)
        if amount_lines:
            lines.append(spacer())
            lines.extend(amount_lines)

    if risk_col:
        risk_lines = _build_segment_risk_summary(df, segment_col, risk_col, top_n=5)
        if risk_lines:
            lines.append(spacer())
            lines.extend(risk_lines)

    lines.extend([
        spacer(),
        subheading("Key Insight"),
        _generate_key_insight(df, segment_col, amount_col, risk_col),
    ])

    return "\n".join(lines)


def answer_segment_question(user_query: str) -> str:
    q = (user_query or "").strip().lower()

    if not q:
        return ""

    df, source_path = _load_best_segment_source()
    if df is None or source_path is None:
        return "Customer segment data is unavailable."

    segment_col = _segment_col(df)
    customer_col = _customer_col(df)
    amount_col = _amount_col(df)
    risk_col = _risk_col(df)

    if segment_col is None:
        return f"Segment data exists, but no segment column was found. Available columns: {list(df.columns)}"

    seg_counts = df[segment_col].fillna("Unknown").astype(str).value_counts(dropna=False)

    if any(x in q for x in ["segment summary", "summarize customer segments", "segment breakdown"]):
        return get_segment_summary()

    if "largest segment" in q or "biggest segment" in q:
        seg = seg_counts.idxmax()
        cnt = seg_counts.max()
        return "\n".join([
            section_title("Largest Segment"),
            spacer(),
            stat_line("Segment", clean_value(seg)),
            stat_line("Records", format_number(cnt)),
        ])

    if "smallest segment" in q:
        seg = seg_counts.idxmin()
        cnt = seg_counts.min()
        return "\n".join([
            section_title("Smallest Segment"),
            spacer(),
            stat_line("Segment", clean_value(seg)),
            stat_line("Records", format_number(cnt)),
        ])

    if "top segments" in q or "show segments" in q:
        lines = [section_title("Top Customer Segments"), spacer()]
        for seg, cnt in seg_counts.head(10).items():
            pct = (cnt / len(df)) * 100 if len(df) else 0
            lines.append(bullet_line(clean_value(seg), f"{format_number(cnt)} ({pct:.2f}%)"))
        return "\n".join(lines)

    if amount_col and any(x in q for x in ["amount", "spend", "highest spend segment", "largest spend segment"]):
        amt = pd.to_numeric(df[amount_col], errors="coerce")
        grouped = (
            df.assign(__amt=amt)
            .groupby(segment_col, dropna=False)["__amt"]
            .sum()
            .sort_values(ascending=False)
        )

        if grouped.empty:
            return "Amount data is unavailable for segment analysis."

        top_seg = grouped.index[0]
        top_amt = grouped.iloc[0]
        return "\n".join([
            section_title("Highest-Spend Segment"),
            spacer(),
            stat_line("Segment", clean_value(top_seg)),
            stat_line("Total amount", _format_amount(top_amt)),
        ])

    if risk_col and any(x in q for x in ["riskiest segment", "highest risk segment", "segment risk"]):
        risk = pd.to_numeric(df[risk_col], errors="coerce")
        grouped = (
            df.assign(__risk=risk)
            .groupby(segment_col, dropna=False)["__risk"]
            .mean()
            .sort_values(ascending=False)
        )

        if grouped.empty:
            return "Risk data is unavailable for segment analysis."

        top_seg = grouped.index[0]
        top_risk = grouped.iloc[0]
        return "\n".join([
            section_title("Highest-Risk Segment"),
            spacer(),
            stat_line("Segment", clean_value(top_seg)),
            stat_line("Average risk score", _format_score(top_risk)),
        ])

    if customer_col and "customer" in q:
        for token in user_query.replace(",", " ").split():
            tok = token.strip().upper()
            if tok.startswith("C"):
                matches = df[df[customer_col].astype(str).str.upper() == tok]
                if not matches.empty:
                    seg_value = matches.iloc[0][segment_col]
                    return "\n".join([
                        section_title("Customer Segment Lookup"),
                        spacer(),
                        stat_line("Customer", tok),
                        stat_line("Segment", clean_value(seg_value)),
                    ])
                return f"Customer {tok} was not found in the segment data."

    return ""
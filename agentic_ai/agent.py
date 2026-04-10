from __future__ import annotations

from agentic_ai.tools.kpi_tool import get_kpi_summary, answer_kpi_question
from agentic_ai.tools.anomaly_tool import get_anomaly_summary, answer_anomaly_question
from agentic_ai.tools.streaming_tool import get_streaming_summary, answer_streaming_question
from agentic_ai.tools.label_tool import get_label_summary, answer_label_question
from agentic_ai.tools.risk_tool import get_risk_summary, answer_risk_question
from agentic_ai.tools.segment_tool import get_segment_summary, answer_segment_question


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _respond_with_fallback(user_query: str, answer_fn, summary_fn) -> str:
    """
    Try the question-specific handler first.
    Fall back to the tool summary only if needed.
    """
    try:
        answer = answer_fn(user_query)
        if answer and str(answer).strip():
            return str(answer).strip()
    except Exception as exc:
        return f"Server error: {exc}"

    try:
        return summary_fn()
    except Exception as exc:
        return f"Server error: {exc}"


def run_agent(user_query: str) -> str:
    if not user_query or not user_query.strip():
        return "Please enter a question."

    q = user_query.strip().lower()

    # ----------------------------
    # ANOMALIES / SUSPICIOUS TRANSACTIONS
    # Keep this before risk because many suspicious/fraud questions
    # are transaction-level anomaly questions, not customer risk summaries.
    # ----------------------------
    anomaly_keywords = [
        "anomaly",
        "anomalies",
        "suspicious transaction",
        "suspicious transactions",
        "suspicious atm",
        "atm anomaly",
        "flagged transaction",
        "flagged transactions",
        "outlier",
        "outliers",
        "show anomalies",
        "show suspicious",
        "why was this flagged",
        "why flagged",
        "top suspicious",
        "top suspicious transactions",
        "highest anomaly",
        "anomaly score",
        "declined transaction",
        "declined transactions",
    ]
    if _has_any(q, anomaly_keywords):
        return _respond_with_fallback(
            user_query,
            answer_anomaly_question,
            get_anomaly_summary,
        )

    # ----------------------------
    # STREAMING / LIVE / LATEST TRANSACTIONS
    # Keep before KPI because "latest" and "transactions" can otherwise
    # be swallowed by broad KPI matching.
    # ----------------------------
    streaming_keywords = [
        "stream",
        "streaming",
        "live transaction",
        "live transactions",
        "latest timestamp",
        "latest transaction",
        "last transaction",
        "last 10 transactions",
        "recent transactions",
        "transaction feed",
        "new transactions",
        "latest batch",
        "recent batch",
        "latest stream",
        "live feed",
    ]
    if _has_any(q, streaming_keywords):
        return _respond_with_fallback(
            user_query,
            answer_streaming_question,
            get_streaming_summary,
        )

    # ----------------------------
    # RISK / HIGH-RISK CUSTOMERS / FRAUD RISK
    # Customer-level risk questions belong here.
    # ----------------------------
    risk_keywords = [
        "risk",
        "risk score",
        "risk overview",
        "high risk",
        "high-risk",
        "risky customer",
        "risky customers",
        "high-risk customer",
        "high-risk customers",
        "riskiest customer",
        "top risk customers",
        "fraud risk",
        "customer risk",
        "risk distribution",
    ]
    if _has_any(q, risk_keywords):
        return _respond_with_fallback(
            user_query,
            answer_risk_question,
            get_risk_summary,
        )

    # ----------------------------
    # SEGMENTS / CUSTOMER GROUPS
    # ----------------------------
    segment_keywords = [
        "segment",
        "segments",
        "customer segment",
        "customer segments",
        "segment summary",
        "segment breakdown",
        "largest segment",
        "smallest segment",
        "segment distribution",
        "customer group",
        "customer groups",
        "mass market",
        "high net worth",
        "high value customers",
        "affluent",
        "business segment",
    ]
    if _has_any(q, segment_keywords):
        return _respond_with_fallback(
            user_query,
            answer_segment_question,
            get_segment_summary,
        )

    # ----------------------------
    # LABELS / TAGS / CLASSIFICATION
    # ----------------------------
    label_keywords = [
        "label",
        "labels",
        "customer labels",
        "classification",
        "classifications",
        "tag",
        "tags",
        "customer category",
        "customer categories",
    ]
    if _has_any(q, label_keywords):
        return _respond_with_fallback(
            user_query,
            answer_label_question,
            get_label_summary,
        )

    # ----------------------------
    # KPI / EXECUTIVE / BUSINESS METRICS
    # Make this narrower so ordinary words like "customers" or "spend"
    # do not capture unrelated questions.
    # ----------------------------
    kpi_keywords = [
        "kpi",
        "kpis",
        "executive summary",
        "executive metrics",
        "business summary",
        "dashboard summary",
        "latest kpi",
        "latest kpis",
        "transaction volume",
        "total transaction volume",
        "total spend",
        "average transaction",
        "average transaction amount",
        "avg amount",
        "transaction count",
        "total customers",
        "customer count",
        "merchant summary",
        "channel summary",
        "geo summary",
        "city has the highest spend",
        "highest spend",
        "top merchant categories",
        "top channels",
    ]
    if _has_any(q, kpi_keywords):
        return _respond_with_fallback(
            user_query,
            answer_kpi_question,
            get_kpi_summary,
        )

    # ----------------------------
    # SMART FALLBACKS FOR COMMON NATURAL QUESTIONS
    # ----------------------------
    if "timestamp" in q or "latest time" in q:
        return _respond_with_fallback(
            user_query,
            answer_streaming_question,
            get_streaming_summary,
        )

    if "flagged" in q or "suspicious" in q:
        return _respond_with_fallback(
            user_query,
            answer_anomaly_question,
            get_anomaly_summary,
        )

    if "riskiest" in q or "risk score" in q:
        return _respond_with_fallback(
            user_query,
            answer_risk_question,
            get_risk_summary,
        )

    if "segment" in q:
        return _respond_with_fallback(
            user_query,
            answer_segment_question,
            get_segment_summary,
        )

    # ----------------------------
    # FALLBACK
    # ----------------------------
    return (
        "I can help with KPI summaries, anomaly detection, risk scores, "
        "customer segments, streaming transactions, and customer labels.\n\n"
        "Try asking:\n"
        "- Show KPI summary\n"
        "- What is the average transaction amount?\n"
        "- Which city has the highest spend?\n"
        "- Show anomalies\n"
        "- Show suspicious ATM transactions\n"
        "- Show anomaly by country\n"
        "- Why was this transaction flagged?\n"
        "- Show high-risk customers\n"
        "- Who is the riskiest customer?\n"
        "- Summarize customer segments\n"
        "- Which segment is the largest?\n"
        "- Show segment breakdown\n"
        "- Show last 10 transactions\n"
        "- What is the latest timestamp?\n"
        "- Show customer labels\n"
        "- Give me executive summary"
    )
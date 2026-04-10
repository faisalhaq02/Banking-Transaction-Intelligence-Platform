from __future__ import annotations

import re

from agentic_ai.tools.kpi_tool import get_kpi_summary
from agentic_ai.tools.segmentation_tool import get_segment_summary
from agentic_ai.tools.anomaly_tool import (
    get_anomaly_summary,
    answer_anomaly_question,
)
from agentic_ai.tools.risk_tool import get_risk_summary
from agentic_ai.tools.spend_prediction_tool import get_spend_prediction_summary
from agentic_ai.tools.investigation_tool import get_priority_investigations

# Optional streaming support
try:
    from agentic_ai.tools.streaming_tool import (
        get_streaming_summary,
        answer_streaming_question,
    )
    STREAMING_AVAILABLE = True
except Exception:
    STREAMING_AVAILABLE = False


def build_agent_response(selected_tool: str, response_text: str) -> str:
    return (
        "Banking Intelligence Agent Response\n"
        f"Selected tool: {selected_tool}\n\n"
        f"{response_text}"
    )


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def help_text() -> str:
    base = (
        "Supported query groups:\n"
        "- KPI: latest kpi, total revenue, total transactions, average transaction value\n"
        "- Segmentation: give me customer segments, segment summary, how many segments do we have\n"
        "- Anomaly: show anomalies, anomaly summary, top anomalies, unusual customers, anomaly reasons, anomalies by channel, anomalies by country, high risk anomalies\n"
        "- Risk: give risk scores, top risky customers, risk distribution, high risk customers\n"
        "- Spend Prediction: who are the top predicted spenders, future spend, highest predicted spend\n"
        "- Investigation: show customers that are both anomalous and high risk, priority investigations\n"
    )

    if STREAMING_AVAILABLE:
        base += (
            "- Streaming: give me streaming summary, latest timestamp, latest transactions, top streaming channels, top merchant categories\n"
        )

    base += (
        "\nExample questions:\n"
        "- latest kpi\n"
        "- what is the total revenue\n"
        "- show customer segments\n"
        "- show anomalies\n"
        "- show top suspicious customers\n"
        "- summarize anomaly reasons\n"
        "- anomalies by country\n"
        "- anomalies by channel\n"
        "- give risk scores\n"
        "- who are the top predicted spenders\n"
        "- show customers that are both anomalous and high risk\n"
    )

    if STREAMING_AVAILABLE:
        base += (
            "- give me streaming summary\n"
            "- give me latest timestamp\n"
            "- show latest streamed transactions\n"
        )

    return base


def route_user_query(user_query: str) -> str:
    try:
        if not user_query or not user_query.strip():
            return build_agent_response(
                selected_tool="none",
                response_text="Please enter a valid query."
            )

        user_query_lower = normalize_text(user_query)

        greeting_phrases = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
        ]

        help_phrases = [
            "help",
            "what can you do",
            "supported queries",
            "how do i use this",
            "how to use this",
            "options",
            "features",
            "available queries",
        ]

        if user_query_lower in greeting_phrases:
            return build_agent_response(
                selected_tool="greeting",
                response_text=(
                    "Hello! I can help with KPI, segmentation, anomaly detection, risk scoring, "
                    "spend prediction, investigation queries"
                    + (", and streaming insights.\n\n" if STREAMING_AVAILABLE else ".\n\n")
                    + help_text()
                ),
            )

        if contains_any(user_query_lower, help_phrases):
            return build_agent_response(
                selected_tool="help",
                response_text=help_text(),
            )

        investigation_phrases = [
            "priority investigation",
            "priority investigations",
            "investigation candidates",
            "both anomalous and high risk",
            "anomalous and high risk",
            "high risk and anomalous",
            "customers needing investigation",
            "customers need investigation",
            "show suspicious high risk customers",
            "suspicious high risk customers",
            "priority customers",
            "who needs investigation",
            "which customers need investigation",
            "show overlap between anomaly and risk",
            "anomaly and risk overlap",
            "risk and anomaly overlap",
            "overlap between risk and anomaly",
        ]

        anomaly_phrases = [
            "anomaly",
            "anomalies",
            "anomalous",
            "outlier",
            "outliers",
            "unusual customer",
            "unusual customers",
            "unusual transactions",
            "highly unusual",
            "suspicious customers",
            "show anomalies",
            "top anomalies",
            "anomaly summary",
            "how many anomalies",
            "fraud",
            "frauds",
            "possible fraud",
            "fraudulent",
            "top suspicious",
            "most suspicious",
            "anomaly reasons",
            "reason summary",
            "why flagged",
            "high risk anomalies",
            "anomalies by channel",
            "anomalies by country",
            "segment anomalies",
        ]

        risk_phrases = [
            "risk",
            "risk score",
            "risk scores",
            "risk summary",
            "high risk",
            "low risk",
            "medium risk",
            "top risky customers",
            "highest risk customers",
            "give risk scores",
            "show risk distribution",
            "customer risk",
            "risky customers",
        ]

        spend_prediction_phrases = [
            "predicted spend",
            "predicted spenders",
            "top predicted spenders",
            "spend prediction",
            "spend predictions",
            "future spend",
            "highest predicted spend",
            "who will spend the most",
            "top spenders",
            "expected spend",
            "forecast spend",
            "spend forecast",
        ]

        segmentation_phrases = [
            "segment",
            "segments",
            "segmentation",
            "customer segment",
            "customer segments",
            "segment summary",
            "how many customer segments",
            "show customer segments",
            "give me customer segments",
            "cluster",
            "clusters",
            "customer groups",
        ]

        kpi_phrases = [
            "kpi",
            "latest kpi",
            "daily kpi",
            "revenue",
            "total revenue",
            "total spend",
            "average transaction value",
            "avg transaction value",
            "transaction value",
            "total transactions",
            "transaction count",
            "business summary",
            "performance summary",
            "dashboard summary",
            "executive summary",
        ]

        streaming_phrases = [
            "stream",
            "streaming",
            "latest timestamp",
            "latest streamed",
            "latest stream",
            "stream summary",
            "streaming summary",
            "latest transaction",
            "latest transactions",
            "recent transactions",
            "stream activity",
            "streamed records",
            "top streaming channels",
            "top merchant categories",
            "top countries in stream",
        ]

        is_investigation = contains_any(user_query_lower, investigation_phrases)
        is_anomaly = contains_any(user_query_lower, anomaly_phrases)
        is_risk = contains_any(user_query_lower, risk_phrases)
        is_spend_prediction = contains_any(user_query_lower, spend_prediction_phrases)
        is_segmentation = contains_any(user_query_lower, segmentation_phrases)
        is_kpi = contains_any(user_query_lower, kpi_phrases)
        is_streaming = STREAMING_AVAILABLE and contains_any(user_query_lower, streaming_phrases)

        # Priority routing
        if is_investigation:
            return build_agent_response(
                selected_tool="investigation",
                response_text=get_priority_investigations(),
            )

        if is_anomaly and is_risk:
            return build_agent_response(
                selected_tool="investigation",
                response_text=get_priority_investigations(),
            )

        if is_streaming:
            try:
                return build_agent_response(
                    selected_tool="streaming",
                    response_text=answer_streaming_question(user_query),
                )
            except Exception:
                return build_agent_response(
                    selected_tool="streaming",
                    response_text=get_streaming_summary(),
                )

        if is_anomaly:
            try:
                return build_agent_response(
                    selected_tool="anomaly",
                    response_text=answer_anomaly_question(user_query),
                )
            except Exception:
                return build_agent_response(
                    selected_tool="anomaly",
                    response_text=get_anomaly_summary(),
                )

        if is_risk:
            return build_agent_response(
                selected_tool="risk",
                response_text=get_risk_summary(),
            )

        if is_spend_prediction:
            return build_agent_response(
                selected_tool="spend_prediction",
                response_text=get_spend_prediction_summary(),
            )

        if is_segmentation:
            return build_agent_response(
                selected_tool="segmentation",
                response_text=get_segment_summary(),
            )

        if is_kpi:
            return build_agent_response(
                selected_tool="kpi",
                response_text=get_kpi_summary(),
            )

        return build_agent_response(
            selected_tool="fallback",
            response_text=(
                "Sorry, I could not match your query to a supported banking intelligence tool.\n\n"
                f"{help_text()}"
            ),
        )

    except Exception as e:
        return build_agent_response(
            selected_tool="error",
            response_text=f'An error occurred while processing the request: "{str(e)}"',
        )
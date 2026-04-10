from __future__ import annotations


def answer_general(query: str) -> str:
    q = query.lower()

    if "what can you do" in q or "help" in q:
        return (
            "I can answer questions about:\n"
            "- customer risk analysis\n"
            "- anomaly / suspicious transaction detection\n"
            "- customer segmentation\n"
            "- executive KPIs\n"
            "- spend prediction\n"
            "- streaming transaction summaries\n"
            "- BI dashboard outputs\n"
            "- pipeline and project overview"
        )

    if "project" in q or "pipeline" in q or "system" in q:
        return (
            "This Banking Transaction Intelligence Platform combines:\n"
            "- streaming transaction ingestion\n"
            "- risk scoring\n"
            "- anomaly detection\n"
            "- customer segmentation\n"
            "- spend prediction\n"
            "- BI exports for dashboards\n"
            "- orchestration through Airflow\n"
            "- optional cloud upload through Azure"
        )

    return (
        "I understood your question, but I do not have a direct answer pattern for it yet.\n\n"
        "Try asking:\n"
        "- Show top anomalous customers\n"
        "- Show top high-risk customers\n"
        "- Summarize customer segments\n"
        "- What are the executive KPIs?\n"
        "- What is happening in streaming?\n"
        "- Show spend prediction summary"
    )
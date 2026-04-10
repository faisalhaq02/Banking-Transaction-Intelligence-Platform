from __future__ import annotations

from agentic_ai.config import TOP_N_DEFAULT
from agentic_ai.utils.cloud_data_access import load_channel_summary_cloud_first


def get_channel_summary(top_n: int = TOP_N_DEFAULT) -> str:
    df = load_channel_summary_cloud_first()

    if df is None:
        return "Channel summary data is unavailable."

    if df.empty:
        return "Channel summary data is available but empty."

    lines = ["Channel summary preview:"]
    for _, row in df.head(top_n).iterrows():
        parts = [f"{col}={row[col]}" for col in df.columns]
        lines.append("- " + ", ".join(parts))

    return "\n".join(lines)
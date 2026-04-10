from __future__ import annotations

from agentic_ai.config import DATA_PATHS
from agentic_ai.utils.data_access import safe_load_parquet, standardize


def get_schema_debug_report() -> str:
    lines = ["Schema debug report:"]

    for name, path in DATA_PATHS.items():
        df = safe_load_parquet(path)
        if df is None:
            lines.append(f"{name}: unavailable at {path}")
            continue

        df = standardize(df)
        lines.append(f"{name}: shape={df.shape}, columns={list(df.columns)}")

    return "\n".join(lines)
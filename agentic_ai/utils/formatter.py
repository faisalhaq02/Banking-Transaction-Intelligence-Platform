from __future__ import annotations


def format_number(value) -> str:
    try:
        if value is None:
            return "N/A"

        value = float(value)

        if abs(value) >= 1_000_000_000:
            return f"{value:,.2f}"
        if abs(value) >= 1_000_000:
            return f"{value:,.2f}"
        if abs(value) >= 1_000:
            return f"{value:,.2f}"
        return f"{value:.2f}"
    except Exception:
        return str(value)
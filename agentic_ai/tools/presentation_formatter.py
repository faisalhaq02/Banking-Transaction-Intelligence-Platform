from __future__ import annotations

from typing import Iterable


def clean_value(value, default: str = "—") -> str:
    try:
        import pandas as pd
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "nan", "null"}:
        return default
    return text


def section_title(title: str) -> str:
    return f"{title}\n" + ("─" * len(title))


def stat_line(label: str, value) -> str:
    return f"{label}: {value}"


def bullet_line(label: str, value) -> str:
    return f"• {label}: {value}"


def subheading(title: str) -> str:
    return title


def spacer() -> str:
    return ""


def join_non_empty(parts: Iterable[str], sep: str = " | ") -> str:
    cleaned = [str(p).strip() for p in parts if p and str(p).strip()]
    return sep.join(cleaned)


def kv(label: str, value, default: str = "—") -> str:
    cleaned = clean_value(value, default=default)
    if cleaned == default:
        return ""
    return f"{label}: {cleaned}"


def numbered_item(index: int, title: str, details: list[str]) -> str:
    title = clean_value(title)
    inline = join_non_empty(details, sep=" | ")
    if inline:
        return f"{index}. {title} | {inline}"
    return f"{index}. {title}"


def simple_item(title: str, details: list[str]) -> str:
    title = clean_value(title)
    inline = join_non_empty(details, sep=" | ")
    if inline:
        return f"• {title} | {inline}"
    return f"• {title}"


def block(title: str, lines: list[str]) -> list[str]:
    output = [section_title(title)]
    output.extend(lines)
    return output
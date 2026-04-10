from __future__ import annotations

from typing import Any, Iterable


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "none", "null", "nan", "na", "n/a", "unknown"}


def clean_value(value: Any, default: str = "Unknown") -> str:
    if _is_missing(value):
        return default
    return str(value).strip()


def clean_text(value: Any, fallback: str = "Unknown") -> str:
    return clean_value(value, default=fallback)


def clean_label(value: Any, fallback: str = "Unknown") -> str:
    text = clean_value(value, default=fallback)
    if text == fallback:
        return fallback

    text = text.replace("_", " ").replace("-", " ").strip()
    if not text:
        return fallback

    words = text.split()
    return " ".join(word.capitalize() for word in words)


def spacer() -> str:
    return ""


def section_title(title: str) -> str:
    return clean_value(title, default="Section")


def subheading(title: str) -> str:
    return clean_value(title, default="Subsection")


def stat_line(label: str, value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    return f"{prefix}{clean_value(label)}: {clean_value(value)}"


def kv_line(key: str, value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    return f"{prefix}{clean_value(key)}: {clean_value(value)}"


def bullet_line(
    primary: Any,
    secondary: Any | None = None,
    indent: int = 0,
    bullet: str = "-",
) -> str:
    prefix = " " * indent
    left = clean_value(primary)
    if secondary is None:
        return f"{prefix}{bullet} {left}"
    return f"{prefix}{bullet} {left}: {clean_value(secondary)}"


def join_lines(lines: Iterable[Any]) -> str:
    output: list[str] = []
    for line in lines:
        if line is None:
            continue
        text = str(line).rstrip()
        if text != "":
            output.append(text)
    return "\n".join(output)


# ------------------------------------------------------------------
# Compatibility wrappers so old imports still work
# ------------------------------------------------------------------
def make_section_title(title: str) -> str:
    return section_title(title)


def make_kv_line(label: str, value: Any) -> str:
    return stat_line(label, value)


def make_empty_message(message: str) -> str:
    return clean_value(message, default="No data available.")


def make_bullet_list(title: str, items: Iterable[Any]) -> str:
    lines = [section_title(title)]
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            lines.append(bullet_line(text))
    return join_lines(lines)


def format_top_items(
    items: Iterable[str],
    heading: str | None = None,
    empty_message: str = "No items found.",
) -> str:
    lines: list[str] = []
    if heading:
        lines.append(section_title(heading))

    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        lines.append(empty_message)
    else:
        lines.extend(cleaned)
    return join_lines(lines)
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def safe_load_parquet(path: str | Path | None):
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def safe_load_csv(path: str | Path | None):
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def safe_load_json(path: str | Path | None):
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return pd.read_json(p)
    except Exception:
        return None


def safe_load_jsonl(path: str | Path | None):
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None


def safe_load_any(path: str | Path | None):
    if path is None:
        return None

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".parquet":
        return safe_load_parquet(p)
    if suffix == ".csv":
        return safe_load_csv(p)
    if suffix == ".json":
        return safe_load_json(p)
    if suffix == ".jsonl":
        return safe_load_jsonl(p)

    for loader in (safe_load_parquet, safe_load_csv, safe_load_jsonl, safe_load_json):
        df = loader(p)
        if df is not None:
            return df
    return None


def safe_load_first_available(paths: Iterable[str | Path | None]):
    for path in paths:
        df = safe_load_any(path)
        if df is not None and not df.empty:
            return df
    return None


def safe_load_latest_file_from_dir(
    directory: str | Path | None,
    patterns: tuple[str, ...] = ("*.parquet", "*.csv", "*.jsonl", "*.json"),
):
    if directory is None:
        return None

    d = Path(directory)
    if not d.exists() or not d.is_dir():
        return None

    files = []
    for pattern in patterns:
        files.extend(d.glob(pattern))

    files = [f for f in files if f.is_file()]
    if not files:
        return None

    latest = max(files, key=lambda f: f.stat().st_mtime)
    return safe_load_any(latest)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in cols:
            return cols[key]
    return None
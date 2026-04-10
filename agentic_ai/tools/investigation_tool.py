from __future__ import annotations

import pandas as pd

from agentic_ai.config import DATA_PATHS, TOP_N_DEFAULT
from agentic_ai.utils.data_access import (
    safe_load_parquet,
    standardize,
    first_matching_column,
)
from agentic_ai.utils.formatter import format_number


def get_priority_investigations(top_n: int = TOP_N_DEFAULT) -> str:
    anomalies = safe_load_parquet(DATA_PATHS["anomaly_scores"])
    risk = safe_load_parquet(DATA_PATHS["risk_scores"])

    if anomalies is None:
        return "Anomaly results are unavailable."

    if risk is None:
        return "Risk scores are unavailable."

    anomalies = standardize(anomalies)
    risk = standardize(risk)

    if anomalies.empty:
        return "Anomaly results are available but empty."

    if risk.empty:
        return "Risk scores are available but empty."

    anom_customer_col = first_matching_column(anomalies, ["customer_id", "cust_id", "customer"])
    risk_customer_col = first_matching_column(risk, ["customer_id", "cust_id", "customer"])

    if not anom_customer_col or not risk_customer_col:
        return "Customer identifier column could not be found in anomaly or risk data."

    anomaly_flag_col = first_matching_column(
        anomalies,
        [
            "ensemble_anomaly_flag",
            "anomaly_flag",
            "iforest_flag",
            "lof_flag",
            "ocsvm_flag",
        ],
    )

    anomaly_score_col = first_matching_column(
        anomalies,
        [
            "ensemble_anomaly_score",
            "anomaly_score",
            "iforest_score",
            "lof_score",
            "ocsvm_score",
        ],
    )

    anomaly_severity_col = first_matching_column(
        anomalies,
        [
            "anomaly_severity",
            "severity",
        ],
    )

    risk_score_col = first_matching_column(
        risk,
        [
            "risk_score",
            "score",
        ],
    )

    risk_label_col = first_matching_column(
        risk,
        [
            "risk_label",
            "risk_bucket",
            "risk_category",
        ],
    )

    total_spend_col = first_matching_column(
        anomalies,
        [
            "total_spend",
            "spend",
        ],
    )

    txn_count_col = first_matching_column(
        anomalies,
        [
            "txn_count",
            "transaction_count",
        ],
    )

    if not anomaly_flag_col:
        return "No anomaly flag column was found in anomaly results."

    if not anomaly_score_col:
        return "No anomaly score column was found in anomaly results."

    if not risk_score_col:
        return "No risk score column was found in risk results."

    if not risk_label_col:
        return "No risk label column was found in risk results."

    anomalies = anomalies.rename(columns={anom_customer_col: "customer_id"})
    risk = risk.rename(columns={risk_customer_col: "customer_id"})

    merged = anomalies.merge(
        risk[
            [
                c
                for c in ["customer_id", risk_score_col, risk_label_col]
                if c in risk.columns
            ]
        ],
        on="customer_id",
        how="inner",
    )

    if merged.empty:
        return "No overlapping customer records were found between anomaly and risk results."

    anomalous_mask = merged[anomaly_flag_col].fillna(0).astype(int) == 1
    high_risk_mask = (
        merged[risk_label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("high", na=False)
    )

    priority = merged[anomalous_mask & high_risk_mask].copy()

    if priority.empty:
        return "No customers were found that are both anomalous and high risk."

    priority = priority.sort_values(
        by=[risk_score_col, anomaly_score_col],
        ascending=[False, False],
    ).head(top_n)

    lines = []
    lines.append(
        f"Found {len(priority)} priority investigation customers that are both anomalous and high risk."
    )
    lines.append("")
    lines.append("Top investigation candidates:")

    for _, row in priority.iterrows():
        customer_id = row["customer_id"]
        anomaly_score = row.get(anomaly_score_col, None)
        risk_score = row.get(risk_score_col, None)
        risk_label = row.get(risk_label_col, "Unknown")
        severity = row.get(anomaly_severity_col, "Unknown") if anomaly_severity_col else "Unknown"
        total_spend = row.get(total_spend_col, None) if total_spend_col else None
        txn_count = row.get(txn_count_col, None) if txn_count_col else None

        details = [
            f"Customer {customer_id}",
            f"anomaly score = {float(anomaly_score):.4f}" if pd.notna(anomaly_score) else "anomaly score = N/A",
            f"risk score = {float(risk_score):.4f}" if pd.notna(risk_score) else "risk score = N/A",
            f"risk label = {risk_label}",
            f"severity = {severity}",
        ]

        if pd.notna(total_spend) if total_spend is not None else False:
            details.append(f"total_spend = {format_number(total_spend)}")

        if pd.notna(txn_count) if txn_count is not None else False:
            details.append(f"txn_count = {int(txn_count)}")

        lines.append("- " + ", ".join(details))

    return "\n".join(lines)
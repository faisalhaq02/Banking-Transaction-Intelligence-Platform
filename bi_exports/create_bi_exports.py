from __future__ import annotations
import os
import json
import pandas as pd

OUTPUT_DIR = '/opt/project/outputs'
PRED_DIR = '/opt/project/outputs/predictions'
EXPORT_DIR = '/opt/project/bi_exports'

os.makedirs(EXPORT_DIR, exist_ok=True)

def load_parquet(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f'[WARN] Missing file: {path}')
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f'[WARN] Failed reading {path}: {e}')
        return None

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def first_match(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

segments = load_parquet(f'{OUTPUT_DIR}/customer_segments.parquet')
anomalies = load_parquet(f'{OUTPUT_DIR}/customer_anomalies.parquet')
risk = load_parquet(f'{OUTPUT_DIR}/customer_risk_scores.parquet')
spend = load_parquet(f'{PRED_DIR}/spend_prediction_outputs.parquet')

if segments is not None:
    segments = standardize(segments)
if anomalies is not None:
    anomalies = standardize(anomalies)
if risk is not None:
    risk = standardize(risk)
if spend is not None:
    spend = standardize(spend)

# -----------------------------
# 1) Executive KPIs
# -----------------------------
customer_level_sources = [df for df in [segments, anomalies, risk, spend] if df is not None]
base = None
for df in customer_level_sources:
    if 'customer_id' in df.columns:
        base = df.copy()
        break

if base is not None:
    total_customers = base['customer_id'].nunique()
    txn_col = first_match(base, ['txn_count'])
    spend_col = first_match(base, ['total_spend', 'predicted_spend'])

    total_transactions = float(base[txn_col].fillna(0).sum()) if txn_col else 0.0
    total_spend = float(base[spend_col].fillna(0).sum()) if spend_col else 0.0
    avg_transaction_value = (total_spend / total_transactions) if total_transactions else 0.0

    exec_df = pd.DataFrame([{
        'total_customers': total_customers,
        'total_transactions': total_transactions,
        'total_spend': total_spend,
        'avg_transaction_value': avg_transaction_value
    }])
    exec_df.to_parquet(f'{EXPORT_DIR}/executive_kpis.parquet', index=False)
    exec_df.to_csv(f'{EXPORT_DIR}/executive_kpis.csv', index=False)
    print('[OK] executive_kpis exported')
else:
    print('[WARN] Could not build executive_kpis')

# -----------------------------
# 2) Segment Summary
# -----------------------------
if segments is not None and 'customer_id' in segments.columns:
    seg_col = first_match(segments, ['predicted_segment', 'segment', 'cluster'])
    seg_name_col = first_match(segments, ['predicted_segment_name', 'segment_name'])
    spend_col = first_match(segments, ['total_spend'])
    txn_col = first_match(segments, ['txn_count'])
    avg_amt_col = first_match(segments, ['avg_amount'])

    if seg_col:
        group_cols = [seg_col]
        if seg_name_col:
            group_cols.append(seg_name_col)

        agg_map = {'customer_id': 'nunique'}
        if spend_col:
            agg_map[spend_col] = 'mean'
        if txn_col:
            agg_map[txn_col] = 'mean'
        if avg_amt_col:
            agg_map[avg_amt_col] = 'mean'

        segment_summary = (
            segments.groupby(group_cols, dropna=False)
            .agg(agg_map)
            .reset_index()
            .rename(columns={
                'customer_id': 'customer_count',
                spend_col if spend_col else '': 'avg_total_spend',
                txn_col if txn_col else '': 'avg_txn_count',
                avg_amt_col if avg_amt_col else '': 'avg_amount'
            })
        )

        segment_summary.to_parquet(f'{EXPORT_DIR}/segment_summary.parquet', index=False)
        segment_summary.to_csv(f'{EXPORT_DIR}/segment_summary.csv', index=False)
        print('[OK] segment_summary exported')
    else:
        print('[WARN] No segment column found')
else:
    print('[WARN] Could not build segment_summary')

# -----------------------------
# 3) Risk + Anomaly Summary
# -----------------------------
merged = None
if anomalies is not None and risk is not None and 'customer_id' in anomalies.columns and 'customer_id' in risk.columns:
    merged = anomalies.merge(risk, on='customer_id', how='outer', suffixes=('_anomaly', '_risk'))
elif anomalies is not None:
    merged = anomalies.copy()
elif risk is not None:
    merged = risk.copy()

if merged is not None:
    cols_to_keep = [c for c in [
        'customer_id',
        'txn_count',
        'total_spend',
        'avg_amount',
        'ensemble_anomaly_flag',
        'ensemble_anomaly_score',
        'anomaly_severity',
        'iforest_flag',
        'iforest_score',
        'lof_flag',
        'lof_score',
        'ocsvm_flag',
        'ocsvm_score',
        'risk_score',
        'risk_bucket',
        'segment_name',
        'customer_segment'
    ] if c in merged.columns]

    risk_anomaly_summary = merged[cols_to_keep].copy()
    risk_anomaly_summary.to_parquet(f'{EXPORT_DIR}/risk_anomaly_summary.parquet', index=False)
    risk_anomaly_summary.to_csv(f'{EXPORT_DIR}/risk_anomaly_summary.csv', index=False)
    print('[OK] risk_anomaly_summary exported')
else:
    print('[WARN] Could not build risk_anomaly_summary')

# -----------------------------
# 4) Spend Prediction Summary
# -----------------------------
if spend is not None:
    pred_col = first_match(spend, ['predicted_spend', 'prediction', 'predicted_value'])
    cols_to_keep = [c for c in [
        'customer_id',
        pred_col,
        'txn_count',
        'total_spend',
        'avg_amount',
        'predicted_segment_name',
        'segment_name',
        'risk_bucket'
    ] if c and c in spend.columns]

    if cols_to_keep:
        spend_summary = spend[cols_to_keep].copy()
        spend_summary.to_parquet(f'{EXPORT_DIR}/spend_prediction_summary.parquet', index=False)
        spend_summary.to_csv(f'{EXPORT_DIR}/spend_prediction_summary.csv', index=False)
        print('[OK] spend_prediction_summary exported')
    else:
        print('[WARN] No usable prediction columns found')
else:
    print('[WARN] Could not build spend_prediction_summary')

print('\nExported files:')
for f in sorted(os.listdir(EXPORT_DIR)):
    print(' -', f)

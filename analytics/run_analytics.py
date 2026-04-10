# analytics/run_analytics.py
# Works inside Airflow Docker container (BASE=/opt/project)
# Also works locally (BASE=.) if you want—just set BASE accordingly.

import os
import duckdb

# Auto-detect base path (Docker vs local)
BASE = "/opt/project" if os.path.exists("/opt/project/data") else "."

con = duckdb.connect()

def show(title: str, query: str):
    print(f"\n=== {title} ===")
    try:
        df = con.execute(query).fetchdf()
        print(df)
    except Exception as e:
        print(f"[ERROR] {title}: {e}")
        raise

def describe(title: str, query: str):
    print(f"\n=== {title} (schema) ===")
    df = con.execute(query).fetchdf()
    print(df)
    return df["column_name"].tolist()

# ---------- Paths (partition-friendly globs) ----------
gold_daily_glob = f"{BASE}/data/gold/daily_kpis/**/*.parquet"
gold_merchant_kpis_glob = f"{BASE}/data/gold/merchant_kpis/**/*.parquet"
gold_customer_features_glob = f"{BASE}/data/gold/customer_features/**/*.parquet"
gold_merchant_risk_glob = f"{BASE}/data/gold/merchant_risk/**/*.parquet"

silver_txn_glob = f"{BASE}/data/silver/transactions/**/*.parquet"

# ---------- Helper: extract event_date from filename (works even if partition columns aren't exposed) ----------
EVENT_DATE_EXPR = """
TRY_CAST(
  regexp_extract(filename, 'event_date=([0-9]{4}-[0-9]{2}-[0-9]{2})', 1)
  AS DATE
)
"""

# =========================
# 1) GOLD: DAILY KPIs
# =========================
# Read with filename=true so we can derive event_date robustly from partition folder name
daily_cols = describe(
    "Gold Daily KPIs",
    f"DESCRIBE SELECT * FROM read_parquet('{gold_daily_glob}')"
)

# Build a safe select list depending on what exists
daily_select = []
# derived event_date from file path
daily_select.append(f"{EVENT_DATE_EXPR} AS event_date")

# include common metrics only if present
for c in ["txn_count", "total_amount", "approval_rate", "high_risk_rate"]:
    if c in daily_cols:
        daily_select.append(c)

# fallback: include all columns if none matched (shouldn't happen, but safe)
if len(daily_select) == 1:
    daily_select.append("*")

show(
    "Daily KPIs",
    f"""
    SELECT {", ".join(daily_select)}
    FROM read_parquet('{gold_daily_glob}', filename=true)
    ORDER BY event_date NULLS LAST
    """
)

# =========================
# 2) SILVER: TRANSACTIONS (Top customers, merchants, approval/decline)
# =========================
try:
    silver_cols = describe(
        "Silver Transactions",
        f"DESCRIBE SELECT * FROM read_parquet('{silver_txn_glob}')"
    )
except Exception as e:
    print(f"\n[WARN] Silver transactions not readable at: {silver_txn_glob}")
    print(f"Reason: {e}")
    silver_cols = []

if silver_cols:
    # Top Customers
    if "customer_id" in silver_cols and "amount" in silver_cols:
        show(
            "Top Customers",
            f"""
            SELECT customer_id, SUM(amount) AS total_spent
            FROM read_parquet('{silver_txn_glob}')
            GROUP BY customer_id
            ORDER BY total_spent DESC
            LIMIT 10
            """
        )
    else:
        print("\n[SKIP] Top Customers: need columns customer_id and amount")

    # Top Merchants
    if "merchant" in silver_cols and "amount" in silver_cols:
        show(
            "Top Merchants",
            f"""
            SELECT merchant, COUNT(*) AS txn, SUM(amount) AS revenue
            FROM read_parquet('{silver_txn_glob}')
            GROUP BY merchant
            ORDER BY revenue DESC
            LIMIT 10
            """
        )
    else:
        print("\n[SKIP] Top Merchants: need columns merchant and amount")

    # Approval vs Decline
    if "status" in silver_cols:
        show(
            "Approval vs Decline",
            f"""
            SELECT status, COUNT(*) AS count
            FROM read_parquet('{silver_txn_glob}')
            GROUP BY status
            ORDER BY count DESC
            """
        )
    else:
        print("\n[SKIP] Approval vs Decline: need column status")

# =========================
# 3) GOLD: OPTIONAL TABLES (merchant_kpis, merchant_risk, customer_features)
#    These are useful Week-5 deliverables. They won't break if missing.
# =========================
def try_preview_table(name: str, glob_path: str, limit: int = 20):
    try:
        cols = describe(name, f"DESCRIBE SELECT * FROM read_parquet('{glob_path}')")
        show(f"{name} (sample)", f"SELECT * FROM read_parquet('{glob_path}') LIMIT {limit}")
        return cols
    except Exception as e:
        print(f"\n[WARN] Skipping {name}: {e}")
        return []

try_preview_table("Gold Merchant KPIs", gold_merchant_kpis_glob)
try_preview_table("Gold Merchant Risk", gold_merchant_risk_glob)
try_preview_table("Gold Customer Features", gold_customer_features_glob)

print("\n✅ Analytics completed successfully.")
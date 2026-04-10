# pipelines/silver_to_gold.py

import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F


# -------------------------
# Config (industry: env-driven, no hardcoded absolute paths)
# -------------------------
BASE = os.environ.get("DATA_BASE", "/opt/project/data")

SILVER_DIR = os.path.join(BASE, "silver", "transactions")  # no wildcard
GOLD_DIR   = os.path.join(BASE, "gold")

GOLD_DAILY    = os.path.join(GOLD_DIR, "daily_kpis")
GOLD_CUSTOMER = os.path.join(GOLD_DIR, "customer_features")
GOLD_MERCHANT = os.path.join(GOLD_DIR, "merchant_kpis")

# Incremental controls
# Set in Airflow: PROCESS_DATE="2026-03-02" 
PROCESS_DATE = os.environ.get("PROCESS_DATE")  # YYYY-MM-DD or None
DATE_FROM    = os.environ.get("DATE_FROM")     # optional range
DATE_TO      = os.environ.get("DATE_TO")       # optional range


def _validate_date(s: str) -> str:
    datetime.strptime(s, "%Y-%m-%d")
    return s


spark = (
    SparkSession.builder
    .appName("silver_to_gold_incremental")
    # Important for partition overwrite behavior with partitioned parquet
    .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# -------------------------
# Read Silver
# -------------------------
silver = spark.read.parquet(SILVER_DIR)

# Ensure event_date exists and is correct
# (your bronze_to_silver writes partitionBy(event_date), but column should also exist)
if "event_date" not in silver.columns:
    # prefer event_date derived from event_ts; fallback ingest_ts if needed
    if "event_ts" in silver.columns:
        silver = silver.withColumn("event_date", F.to_date("event_ts"))
    else:
        silver = silver.withColumn("event_date", F.to_date(F.col("ingest_ts")))

# Basic schema guardrails (industry: fail fast)
required = {"event_date", "amount", "status", "customer_id", "merchant_id"}
missing = required - set(silver.columns)
if missing:
    raise Exception(f"Missing required columns in silver: {sorted(missing)}")

# -------------------------
# Incremental filtering
# -------------------------
if PROCESS_DATE:
    d = _validate_date(PROCESS_DATE)
    silver = silver.filter(F.col("event_date") == F.lit(d))
elif DATE_FROM and DATE_TO:
    d1 = _validate_date(DATE_FROM)
    d2 = _validate_date(DATE_TO)
    silver = silver.filter((F.col("event_date") >= F.lit(d1)) & (F.col("event_date") <= F.lit(d2)))
else:
    # If no date controls provided, process the whole dataset (ok for small class data)
    pass

# If the filter yields no rows, exit cleanly (industry: no empty overwrite)
if silver.limit(1).count() == 0:
    print("No silver rows found for given date filter; skipping gold write.")
    spark.stop()
    raise SystemExit(0)

# -------------------------
# 1) Daily KPIs (partitioned by event_date)
# -------------------------
daily = (
    silver.groupBy("event_date")
    .agg(
        F.count("*").alias("txn_count"),
        F.sum("amount").alias("total_amount"),
        F.avg(F.when(F.col("status") == "APPROVED", 1).otherwise(0)).alias("approval_rate"),
        # Handle missing is_high_risk gracefully
        F.avg(F.when(F.coalesce(F.col("is_high_risk"), F.lit(False)) == True, 1).otherwise(0)).alias("high_risk_rate"),
    )
)

(daily.write
 .mode("overwrite")
 .partitionBy("event_date")
 .parquet(GOLD_DAILY)
)

# -------------------------
# 2) Customer Features (incremental)
# Industry note: true customer_features are often maintained via MERGE/UPSERT in Delta/Iceberg.
# For parquet, we commonly compute "features_by_day" or "features_windowed".
# We'll do features_by_day (partitioned by event_date) to keep it incremental and idempotent.
# -------------------------
cust = (
    silver.groupBy("event_date", "customer_id")
    .agg(
        F.count("*").alias("txn_count"),
        F.sum("amount").alias("total_spend"),
        F.avg("amount").alias("avg_amount"),
        F.max("amount").alias("max_amount"),
    )
)

(cust.write
 .mode("overwrite")
 .partitionBy("event_date")
 .parquet(GOLD_CUSTOMER)
)

# -------------------------
# 3) Merchant KPIs (incremental, partitioned)
# -------------------------
merchants = (
    silver.groupBy("event_date", "merchant_id")
    .agg(
        F.count("*").alias("txn_count"),
        F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_amount"),
        F.avg(F.when(F.coalesce(F.col("is_high_risk"), F.lit(False)) == True, 1).otherwise(0)).alias("high_risk_rate"),
        F.avg(F.when(F.col("status") == "APPROVED", 1).otherwise(0)).alias("approval_rate"),
    )
)

(merchants.write
 .mode("overwrite")
 .partitionBy("event_date")
 .parquet(GOLD_MERCHANT)
)

# -------------------------
# Minimal summary (avoid expensive full counts)
# -------------------------
print("=== Silver->Gold Incremental Write Completed ===")
if PROCESS_DATE:
    print(f"Processed date: {PROCESS_DATE}")
elif DATE_FROM and DATE_TO:
    print(f"Processed range: {DATE_FROM} to {DATE_TO}")
else:
    print("Processed: FULL dataset (no date filter)")

spark.stop()
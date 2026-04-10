import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# Base data directory inside Docker containers (repo mounted to /opt/project)
BASE = os.environ.get("DATA_BASE", "/opt/project/data")

BRONZE_PATH = os.path.join(BASE, "bronze", "transactions", "*")
SILVER_PATH = os.path.join(BASE, "silver", "transactions")

DLQ_TOPIC = "transactions_dlq"
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")

# Tuning knobs (safe defaults for laptop / docker)
SHUFFLE_PARTITIONS = os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16")
SILVER_PARTITIONS = int(os.environ.get("SILVER_PARTITIONS", "16"))

spark = (
    SparkSession.builder
    .appName("bronze_to_silver_with_dlq")
    .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.sql.files.maxRecordsPerFile", "500000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# -----------------------
# Read Bronze
# -----------------------
bronze = spark.read.parquet(BRONZE_PATH)

# Normalize + derive event time/date (always present)
base = (
    bronze
    .withColumn("event_ts", F.to_timestamp("timestamp"))
    .withColumn("event_date", F.to_date(F.coalesce(F.col("event_ts"), F.col("ingest_ts"))))
    .withColumn("currency", F.upper(F.trim(F.col("currency"))))
    .withColumn("channel",  F.upper(F.trim(F.col("channel"))))
    .withColumn("country",  F.upper(F.trim(F.col("country"))))
    .withColumn("status",   F.upper(F.trim(F.col("status"))))
)

# -----------------------
# Validation
# -----------------------
invalid_cond = (
    F.col("transaction_id").isNull() |
    F.col("amount").isNull() |
    (F.col("amount") < 0) |
    F.col("timestamp").isNull() |
    F.col("event_date").isNull()
)

valid = base.filter(~invalid_cond)
invalid = base.filter(invalid_cond)

# -----------------------
# Counts (ONE Spark job for bronze/valid/invalid)
#   - avoids cache/persist OOM
#   - avoids 3-4 full scans
# -----------------------
counts_row = (
    base.select(F.when(invalid_cond, F.lit("invalid")).otherwise(F.lit("valid")).alias("tag"))
        .groupBy("tag")
        .count()
)

# Convert grouped result to dict with defaults
counts = {r["tag"]: int(r["count"]) for r in counts_row.collect()}
valid_cnt = counts.get("valid", 0)
invalid_cnt = counts.get("invalid", 0)
bronze_cnt = valid_cnt + invalid_cnt

# -----------------------
# DLQ write (only if invalid exists)
#   - uses invalid_cnt computed above (no extra action)
# -----------------------
if invalid_cnt > 0:
    dlq_df = invalid.select(
        F.lit(None).cast("string").alias("key"),
        F.to_json(
            F.struct(
                F.col("*"),
                F.lit("VALIDATION_FAILED").alias("error_reason"),
                F.current_timestamp().alias("dlq_ts"),
            )
        ).alias("value")
    )

    (dlq_df.write
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP)
        .option("topic", DLQ_TOPIC)
        .save()
    )

# -----------------------
# Deduplicate -> Silver
# keep latest ingest_ts for each transaction_id
# -----------------------
w = Window.partitionBy("transaction_id").orderBy(F.col("ingest_ts").desc_nulls_last())

silver = (
    valid
    .withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# Repartition by event_date for partitioned write
silver_out = silver.repartition(SILVER_PARTITIONS, "event_date")

(silver_out.write
    .mode("overwrite")
    .partitionBy("event_date")
    .parquet(SILVER_PATH)
)

# Silver count (separate action; dedupe changes row count)
silver_cnt = silver.count()

print("=== Bronze -> Silver Summary ===")
print("Bronze rows:  ", bronze_cnt)
print("Valid rows:   ", valid_cnt)
print("Invalid rows: ", invalid_cnt)
print("Silver rows:  ", silver_cnt)

spark.stop()


from pyspark.sql import SparkSession, functions as F, types as T

BRONZE_PATH = "data/bronze/transactions"
SILVER_PATH = "data/silver/transactions"
CHECKPOINT = "checkpoints/bronze_to_silver"

KAFKA_BOOTSTRAP = "localhost:9092"
DLQ_TOPIC = "transactions_dlq"

# Same schema you saw in bronze (adjust if you add fields)
schema = T.StructType([
    T.StructField("transaction_id", T.StringType()),
    T.StructField("timestamp", T.StringType()),
    T.StructField("customer_id", T.StringType()),
    T.StructField("merchant_id", T.StringType()),
    T.StructField("amount", T.DoubleType()),
    T.StructField("currency", T.StringType()),
    T.StructField("channel", T.StringType()),
    T.StructField("city", T.StringType()),
    T.StructField("country", T.StringType()),
    T.StructField("device_id", T.StringType()),
    T.StructField("status", T.StringType()),
    T.StructField("is_high_risk", T.BooleanType()),
    T.StructField("risk_reason", T.StringType()),
    T.StructField("json_str", T.StringType()),
    T.StructField("ingest_ts", T.TimestampType()),
    T.StructField("hour", T.IntegerType()),
    # event_date may exist if Kafka->Bronze stream created it; handle both cases below
    T.StructField("event_date", T.DateType()),
])

spark = SparkSession.builder.appName("bronze_to_silver_stream").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Read Bronze as a STREAM (it watches for new parquet files)
bronze = (spark.readStream
  .format("parquet")
  .schema(schema)            # important for streaming parquet
  .load(BRONZE_PATH)
)

# If event_date is missing/null, derive it from timestamp
bronze = (bronze
  .withColumn("event_ts", F.to_timestamp("timestamp"))
  .withColumn("event_date", F.coalesce(F.col("event_date"), F.to_date("event_ts")))
  .withColumn("hour", F.coalesce(F.col("hour"), F.hour("event_ts")))
)

# ---- Validation rules (customize freely) ----
invalid_reason = F.when(F.col("transaction_id").isNull(), F.lit("missing_transaction_id")) \
    .when(F.col("event_ts").isNull(), F.lit("bad_timestamp")) \
    .when(F.col("amount").isNull(), F.lit("missing_amount")) \
    .when(F.col("amount") < 0, F.lit("negative_amount")) \
    .when(~F.col("status").isin("APPROVED", "DECLINED"), F.lit("invalid_status")) \
    .otherwise(F.lit(None))

with_flags = bronze.withColumn("invalid_reason", invalid_reason)
bad = with_flags.filter(F.col("invalid_reason").isNotNull())
good = with_flags.filter(F.col("invalid_reason").isNull())

# ---- Dedup (exactly what “week 3” expects) ----
# watermark prevents state from growing forever
good_dedup = (good
  .withWatermark("event_ts", "2 hours")
  .dropDuplicates(["transaction_id"])
)

# ---- Write GOOD -> Silver (Parquet) ----
silver_query = (good_dedup
  .drop("invalid_reason")
  .writeStream
  .format("parquet")
  .outputMode("append")
  .option("path", SILVER_PATH)
  .option("checkpointLocation", CHECKPOINT + "/silver")
  .partitionBy("event_date", "hour")
  .trigger(processingTime="30 seconds")
  .start()
)

# ---- Write BAD -> Kafka DLQ ----
dlq_df = (bad.select(
    F.col("transaction_id").cast("string").alias("key"),
    F.to_json(F.struct(
        F.col("invalid_reason").alias("reason"),
        F.col("json_str").alias("raw_json"),
        *[F.col(c) for c in bad.columns if c not in ["invalid_reason"]]
    )).alias("value")
))

dlq_query = (dlq_df.writeStream
  .format("kafka")
  .outputMode("append")
  .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
  .option("topic", DLQ_TOPIC)
  .option("checkpointLocation", CHECKPOINT + "/dlq")
  .trigger(processingTime="30 seconds")
  .start()
)

silver_query.awaitTermination()
dlq_query.awaitTermination()

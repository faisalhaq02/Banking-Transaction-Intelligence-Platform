from pyspark.sql import SparkSession, functions as F, types as T

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"

BRONZE_PATH = "data/bronze/transactions"
CHECKPOINT = "checkpoints/kafka_to_bronze"

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
])

spark = (SparkSession.builder
  .appName("kafka_to_bronze_stream")
  .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

raw = (spark.readStream
  .format("kafka")
  .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
  .option("subscribe", TOPIC)
  .option("startingOffsets", "earliest")   # IMPORTANT for first run
  .load())

parsed = (raw
  .select(F.col("value").cast("string").alias("json_str"))
  .withColumn("data", F.from_json("json_str", schema))
  .select("json_str", "data.*")
  .withColumn("ingest_ts", F.current_timestamp())
  .withColumn("event_ts", F.to_timestamp("timestamp"))
  .withColumn("event_date", F.to_date("event_ts"))
  .withColumn("hour", F.hour("event_ts"))
)

query = (parsed.writeStream
  .format("parquet")
  .outputMode("append")
  .option("path", BRONZE_PATH)
  .option("checkpointLocation", CHECKPOINT)
  .partitionBy("event_date", "hour")
    .trigger(availableNow=True)
  .start()
)

query.awaitTermination()

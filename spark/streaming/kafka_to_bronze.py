from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_date, date_format, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType

TOPIC = "bank_txn_stream"
KAFKA_BOOTSTRAP = "kafka:29092"  # docker network address

BRONZE_PATH = "/opt/spark-data/bronze/transactions"
CHECKPOINT_PATH = "/opt/spark-data/checkpoints/bronze_txn"

schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("merchant_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("currency", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("city", StringType(), True),
    StructField("country", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("status", StringType(), True),

    # Optional fields (if you add them later)
    StructField("is_high_risk", BooleanType(), True),
    StructField("risk_reason", StringType(), True),
])

def main():
    spark = (
        SparkSession.builder
        .appName("KafkaToBronzeTransactions")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    json_df = raw.select(col("value").cast("string").alias("json_str"))

    parsed = (
        json_df
        .select(from_json(col("json_str"), schema).alias("data"), col("json_str"))
        .select("data.*", "json_str")
    )

    enriched = (
        parsed
        .withColumn("ingest_ts", current_timestamp())
        .withColumn("date", to_date(col("ingest_ts")))
        .withColumn("hour", date_format(col("ingest_ts"), "HH"))
    )

    good = enriched.filter(col("transaction_id").isNotNull())
    bad = enriched.filter(col("transaction_id").isNull())

    good_query = (
        good.writeStream
        .format("parquet")
        .option("path", BRONZE_PATH)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .partitionBy("date", "hour")
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .start()
    )

    bad_query = (
        bad.writeStream
        .format("json")
        .option("path", "/opt/spark-data/bronze/bad_records")
        .option("checkpointLocation", "/opt/spark-data/checkpoints/bad_records")
        .outputMode("append")
        .start()
    )

    print("✅ Spark streaming started. Writing Bronze Parquet to:", BRONZE_PATH)
    good_query.awaitTermination()

if __name__ == "__main__":
    main()

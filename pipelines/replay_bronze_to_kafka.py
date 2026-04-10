from pyspark.sql import SparkSession, functions as F

BRONZE_PATH = "data/bronze/transactions/*"
BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"

spark = SparkSession.builder.appName("replay_bronze_to_kafka").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

bronze = spark.read.parquet(BRONZE_PATH)

kafka_df = bronze.select(
    F.col("transaction_id").cast("string").alias("key"),
    F.to_json(F.struct(*[F.col(c) for c in bronze.columns])).alias("value")
)

(kafka_df.write
  .format("kafka")
  .option("kafka.bootstrap.servers", BOOTSTRAP)
  .option("topic", TOPIC)
  .save()
)

print(f"Replayed {bronze.count()} rows from Bronze to Kafka topic '{TOPIC}'")

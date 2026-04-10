from pyspark.sql import SparkSession, functions as F

SILVER_PATH = "data/silver/transactions/*"
GOLD_MERCHANT = "data/gold/merchant_risk"

spark = SparkSession.builder.appName("silver_to_gold_merchants").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

silver = spark.read.parquet(SILVER_PATH)

merchants = (
    silver.groupBy("merchant_id")
    .agg(
        F.count("*").alias("txn_count"),
        F.avg(F.when(F.col("is_high_risk") == True, 1).otherwise(0)).alias("high_risk_rate"),
        F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_amount")
    )
)

merchants.write.mode("overwrite").parquet(GOLD_MERCHANT)

print("Merchant gold created:", merchants.count())

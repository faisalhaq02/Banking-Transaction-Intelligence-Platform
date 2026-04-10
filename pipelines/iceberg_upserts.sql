CREATE NAMESPACE IF NOT EXISTS lakehouse.banking;

DROP TABLE IF EXISTS lakehouse.banking.daily_kpis;
CREATE TABLE lakehouse.banking.daily_kpis
USING iceberg
PARTITIONED BY (days(event_date))
AS
SELECT * FROM parquet.`/opt/project/data/gold/daily_kpis`;

DROP TABLE IF EXISTS lakehouse.banking.customer_features;
CREATE TABLE lakehouse.banking.customer_features
USING iceberg
PARTITIONED BY (days(event_date))
AS
SELECT * FROM parquet.`/opt/project/data/gold/customer_features`;

DROP TABLE IF EXISTS lakehouse.banking.merchant_risk;
CREATE TABLE lakehouse.banking.merchant_risk
USING iceberg
AS
SELECT * FROM parquet.`/opt/project/data/gold/merchant_risk`;

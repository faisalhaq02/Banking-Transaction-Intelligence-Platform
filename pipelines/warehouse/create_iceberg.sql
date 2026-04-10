-- =========================
-- Iceberg Lakehouse (Banking)
-- =========================

-- Use a namespace (database)
CREATE NAMESPACE IF NOT EXISTS lakehouse.banking;

-- ---------- DAILY KPIs ----------
CREATE TABLE IF NOT EXISTS lakehouse.banking.daily_kpis
USING iceberg
PARTITIONED BY (days(kpi_date))
AS
SELECT * FROM parquet.`/opt/project/data/gold/daily_kpis`
WHERE 1 = 0;

MERGE INTO lakehouse.banking.daily_kpis t
USING (
  SELECT * FROM parquet.`/opt/project/data/gold/daily_kpis`
) s
ON t.kpi_date = s.kpi_date
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;


-- ---------- CUSTOMER FEATURES ----------
CREATE TABLE IF NOT EXISTS lakehouse.banking.customer_features
USING iceberg
PARTITIONED BY (days(feature_date))
AS
SELECT * FROM parquet.`/opt/project/data/gold/customer_features`
WHERE 1 = 0;

MERGE INTO lakehouse.banking.customer_features t
USING (
  SELECT * FROM parquet.`/opt/project/data/gold/customer_features`
) s
ON t.customer_id = s.customer_id
AND t.feature_date = s.feature_date
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;


-- ---------- MERCHANT KPIs ----------
CREATE TABLE IF NOT EXISTS lakehouse.banking.merchant_kpis
USING iceberg
PARTITIONED BY (days(kpi_date))
AS
SELECT * FROM parquet.`/opt/project/data/gold/merchant_kpis`
WHERE 1 = 0;

MERGE INTO lakehouse.banking.merchant_kpis t
USING (
  SELECT * FROM parquet.`/opt/project/data/gold/merchant_kpis`
) s
ON t.merchant_id = s.merchant_id
AND t.kpi_date = s.kpi_date
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;
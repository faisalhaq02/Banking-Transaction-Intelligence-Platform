from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.datasets import Dataset
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from azure.storage.blob import BlobServiceClient

# ensure local dags imports work
import sys
sys.path.append(str(Path(__file__).parent))


def run_load_blob_to_sql():
    from load_blob_to_sql import main
    main()

# ----------------------------
# Datasets
# ----------------------------
SILVER_DATASET = Dataset("file:///opt/project/data/silver/transactions")
GOLD_DATASET = Dataset("file:///opt/project/data/gold")
ICEBERG_DATASET = Dataset("file:///opt/project/data/iceberg/warehouse")
MODEL_DATASET = Dataset("file:///opt/project/models")

# ----------------------------
# Defaults
# ----------------------------
default_args = {
    "owner": "team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
}

SPARK_MASTER_URL = "spark://spark-master:7077"
ICEBERG_VERSION = "1.10.1"
ICEBERG_PACKAGE = f"org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:{ICEBERG_VERSION}"

PROJECT_ROOT = Path("/opt/project")
STREAM_BATCH_DIR = PROJECT_ROOT / "data" / "stream_batches"

AZURE_CONNECTION_STRING_ENV = "AZURE_STORAGE_CONNECTION_STRING"
AZURE_BRONZE_CONTAINER_ENV = "CONTAINER_NAME"
AZURE_BRONZE_PREFIX = "streaming_transactions"
AZURE_SYNC_MAX_FILES = 20


def sync_latest_bronze_from_azure(**context):
    connection_string = os.getenv(AZURE_CONNECTION_STRING_ENV)
    container_name = os.getenv(AZURE_BRONZE_CONTAINER_ENV, "bronze")

    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")

    STREAM_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    blobs = [
        blob
        for blob in container_client.list_blobs(name_starts_with=AZURE_BRONZE_PREFIX)
        if blob.name.endswith(".jsonl")
    ]

    if not blobs:
        raise FileNotFoundError(
            f"No Azure bronze blobs found in container '{container_name}' "
            f"with prefix '{AZURE_BRONZE_PREFIX}'"
        )

    blobs.sort(key=lambda b: b.last_modified, reverse=True)
    latest_blobs = blobs[:AZURE_SYNC_MAX_FILES]

    downloaded = 0
    skipped = 0

    for blob in latest_blobs:
        local_file = STREAM_BATCH_DIR / Path(blob.name).name

        if local_file.exists() and local_file.stat().st_size == blob.size:
            print(f"Skipping existing local file: {local_file}")
            skipped += 1
            continue

        print(f"Downloading Azure blob {blob.name} -> {local_file}")
        with local_file.open("wb") as f:
            download_stream = container_client.download_blob(blob.name)
            f.write(download_stream.readall())

        downloaded += 1

    print(
        f"Azure bronze sync complete. "
        f"Downloaded={downloaded}, skipped={skipped}, checked={len(latest_blobs)}"
    )


def get_latest_stream_batch_file(**context):
    if not STREAM_BATCH_DIR.exists():
        raise FileNotFoundError(f"Stream batch directory not found: {STREAM_BATCH_DIR}")

    batch_files = sorted(
        STREAM_BATCH_DIR.glob("transactions_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not batch_files:
        raise FileNotFoundError(f"No stream batch files found in {STREAM_BATCH_DIR}")

    latest_file = batch_files[0]
    print(f"Latest stream batch file found: {latest_file}")
    context["ti"].xcom_push(key="latest_stream_batch_file", value=str(latest_file))


with DAG(
    dag_id="banking_intelligence_platform",
    default_args=default_args,
    start_date=datetime(2026, 2, 1),
    schedule="@hourly",
    catchup=False,
    max_active_runs=1,
    concurrency=4,
    dagrun_timeout=timedelta(minutes=45),
    tags=["banking", "lakehouse", "wanalytics", "ml", "iceberg", "azure"],
    params={
        "spark_shuffle_partitions": Param(64, type="integer", minimum=8, maximum=2000),
        "min_silver_bronze_ratio": Param(0.70, type="number", minimum=0.0, maximum=1.0),
    },
) as dag:

    # ----------------------------
    # Azure Bronze -> Local stream_batches
    # ----------------------------
    sync_latest_bronze_from_azure_task = PythonOperator(
        task_id="sync_latest_bronze_from_azure",
        python_callable=sync_latest_bronze_from_azure,
        execution_timeout=timedelta(minutes=10),
    )

    get_latest_stream_batch_file_task = PythonOperator(
        task_id="get_latest_stream_batch_file",
        python_callable=get_latest_stream_batch_file,
        execution_timeout=timedelta(minutes=5),
    )

    # ----------------------------
    # Bronze -> Silver
    # ----------------------------
    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=f"""
        set -euo pipefail

        export DATA_BASE="/opt/project/data"
        export KAFKA_BOOTSTRAP="kafka:9092"
        export SPARK_SHUFFLE_PARTITIONS="16"
        export SILVER_PARTITIONS="16"

        /opt/spark/bin/spark-submit \
          --master {SPARK_MASTER_URL} \
          --conf spark.sql.shuffle.partitions=16 \
          --conf spark.sql.adaptive.enabled=true \
          --conf spark.sql.adaptive.coalescePartitions.enabled=true \
          --executor-memory 1g \
          --driver-memory 1g \
          --name bronze_to_silver \
          /opt/project/pipelines/bronze_to_silver.py
        """,
        execution_timeout=timedelta(minutes=20),
        outlets=[SILVER_DATASET],
    )

    validate_silver = BashOperator(
        task_id="validate_silver",
        bash_command="""
        set -euo pipefail
        test -d /opt/project/data/silver/transactions
        test "$(find /opt/project/data/silver/transactions -name '*.parquet' | wc -l)" -gt 0
        """,
        execution_timeout=timedelta(minutes=5),
    )

    # ----------------------------
    # Silver -> Gold
    # ----------------------------
    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=f"""
        set -euo pipefail
        export DATA_BASE="/opt/project/data"

        /opt/spark/bin/spark-submit \
          --master {SPARK_MASTER_URL} \
          --conf spark.sql.adaptive.enabled=true \
          --conf spark.sql.adaptive.coalescePartitions.enabled=true \
          --conf spark.sql.shuffle.partitions={{{{ params.spark_shuffle_partitions }}}} \
          --executor-memory 4g \
          --driver-memory 4g \
          --name silver_to_gold \
          /opt/project/pipelines/silver_to_gold.py
        """,
        execution_timeout=timedelta(minutes=20),
        outlets=[GOLD_DATASET],
    )

    validate_gold = BashOperator(
        task_id="validate_gold",
        bash_command="""
        set -euo pipefail
        test -d /opt/project/data/gold
        test "$(find /opt/project/data/gold -name '*.parquet' | wc -l)" -gt 0
        """,
        execution_timeout=timedelta(minutes=5),
    )

    # ----------------------------
    # Week 5: ML model training
    # ----------------------------
    train_customer_segmentation = BashOperator(
        task_id="train_customer_segmentation",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/models /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python -c "import pandas, pyarrow, sklearn, joblib; print('Segmentation libs OK')"
        /home/airflow/.local/bin/python /opt/project/ml/train_customer_segmentation.py

        test -f /opt/project/models/customer_segmentation_kmeans.pkl
        test -f /opt/project/models/customer_segmentation_scaler.pkl
        test -f /opt/project/models/customer_segmentation_metrics.json
        test -f /opt/project/outputs/customer_segments.parquet
        test -f /opt/project/outputs/segment_profiles.csv
        """,
        execution_timeout=timedelta(minutes=15),
        outlets=[MODEL_DATASET],
    )

    segmentation_pca_visualization = BashOperator(
        task_id="segmentation_pca_visualization",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/models /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python -c "import pandas, pyarrow, sklearn, joblib, matplotlib; print('PCA libs OK')"
        /home/airflow/.local/bin/python /opt/project/ml/segmentation_pca_visualization.py

        test -f /opt/project/outputs/pca_customer_segments.png
        test -f /opt/project/outputs/customer_segments_pca.parquet
        test -f /opt/project/outputs/pca_explained_variance.json
        test -f /opt/project/models/customer_segmentation_pca.pkl
        """,
        execution_timeout=timedelta(minutes=10),
        outlets=[MODEL_DATASET],
    )

    segmentation_feature_profiles = BashOperator(
        task_id="segmentation_feature_profiles",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/outputs /opt/project/models
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python -c "import pandas, pyarrow, matplotlib, sklearn; print('Feature profile libs OK')"
        /home/airflow/.local/bin/python /opt/project/ml/segmentation_feature_profiles.py

        test -f /opt/project/outputs/segment_feature_profiles.csv
        test -f /opt/project/outputs/segment_feature_profiles_zscore.csv
        test -f /opt/project/outputs/segment_top_features.json
        test -f /opt/project/outputs/segment_feature_heatmap.png
        """,
        execution_timeout=timedelta(minutes=10),
        outlets=[MODEL_DATASET],
    )

    detect_customer_anomalies = BashOperator(
        task_id="detect_customer_anomalies",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/models /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python -c "import pandas, pyarrow, sklearn, joblib; print('Multi-model anomaly libs OK')"
        /home/airflow/.local/bin/python /opt/project/ml/detect_customer_anomalies.py

        test -f /opt/project/outputs/customer_anomalies.parquet
        test -f /opt/project/outputs/top_unusual_customers.csv
        test -f /opt/project/outputs/anomaly_summary.json
        test -f /opt/project/models/anomaly_isolation_forest.pkl
        test -f /opt/project/models/anomaly_lof.pkl
        test -f /opt/project/models/anomaly_ocsvm.pkl
        test -f /opt/project/models/anomaly_scaler.pkl
        """,
        execution_timeout=timedelta(minutes=12),
        outlets=[MODEL_DATASET],
    )

    generate_risk_scores = BashOperator(
        task_id="generate_risk_scores",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/outputs /opt/project/models
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python -c "import pandas, pyarrow; print('Risk scoring libs OK')"
        /home/airflow/.local/bin/python /opt/project/ml/generate_risk_scores.py

        test -f /opt/project/outputs/customer_risk_scores.parquet
        test -f /opt/project/outputs/top_high_risk_customers.csv
        test -f /opt/project/outputs/risk_bucket_summary.json
        """,
        execution_timeout=timedelta(minutes=10),
        outlets=[MODEL_DATASET],
    )
    
    enrich_risk_outputs = BashOperator(
    task_id="enrich_risk_outputs",
    bash_command="""
    set -euo pipefail

    mkdir -p /opt/project/outputs
    unset PYTHONNOUSERSITE || true
    export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

    /home/airflow/.local/bin/python -c "import pandas, pyarrow, numpy; print('Risk enrichment libs OK')"
    /home/airflow/.local/bin/python /opt/project/ml/enrich_risk_outputs.py

    test -f /opt/project/outputs/customer_risk_scores.parquet
    test -f /opt/project/outputs/customer_risk_scores_enriched.csv
    """,
    execution_timeout=timedelta(minutes=6),
    outlets=[MODEL_DATASET],
    )

    train_spend_prediction = BashOperator(
        task_id="train_spend_prediction",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/models /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python -c "import pandas, pyarrow, sklearn, joblib, matplotlib; print('Spend prediction libs OK')"
        /home/airflow/.local/bin/python /opt/project/ml/train_spend_prediction.py

        test -f /opt/project/models/spend_prediction_random_forest.pkl
        test -f /opt/project/outputs/spend_prediction_metrics.json
        test -f /opt/project/outputs/spend_predictions.parquet
        test -f /opt/project/outputs/spend_feature_importance.csv
        test -f /opt/project/outputs/spend_feature_importance.png
        """,
        execution_timeout=timedelta(minutes=12),
        outlets=[MODEL_DATASET],
    )

    model_monitoring = BashOperator(
        task_id="model_monitoring",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python -c "import pandas, pyarrow; print('Monitoring libs OK')"
        /home/airflow/.local/bin/python /opt/project/ml/model_monitoring.py

        test -f /opt/project/outputs/model_monitoring.json
        test -f /opt/project/outputs/ml_run_history.csv
        """,
        execution_timeout=timedelta(minutes=8),
        outlets=[MODEL_DATASET],
    )

    model_registry = BashOperator(
        task_id="model_registry",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/models/registry /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python /opt/project/ml/model_registry.py

        test -f /opt/project/outputs/model_registry.json
        """,
        execution_timeout=timedelta(minutes=8),
        outlets=[MODEL_DATASET],
    )

    run_customer_segmentation_inference = BashOperator(
        task_id="run_customer_segmentation_inference",
        bash_command=(
            "/home/airflow/.local/bin/python "
            "/opt/project/ml/predict_with_registry.py "
            "--model customer_segmentation "
            "--input /opt/project/data/gold/customer_features "
            "--output-dir /opt/project/outputs/predictions"
        ),
    )

    run_spend_prediction_inference = BashOperator(
        task_id="run_spend_prediction_inference",
        bash_command=(
            "/home/airflow/.local/bin/python "
            "/opt/project/ml/predict_with_registry.py "
            "--model spend_prediction "
            "--input /opt/project/outputs/predictions/customer_segmentation_predictions.parquet "
            "--output-dir /opt/project/outputs/predictions"
        ),
    )

    validate_ml_outputs = BashOperator(
        task_id="validate_ml_outputs",
        bash_command=(
            "test -f /opt/project/outputs/predictions/customer_segmentation_predictions.parquet "
            "&& test -f /opt/project/outputs/predictions/spend_prediction_outputs.parquet "
            "&& echo 'ML prediction outputs validated successfully.'"
        ),
    )

    data_drift_detection = BashOperator(
        task_id="data_drift_detection",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python /opt/project/ml/data_drift_detection.py

        test -f /opt/project/outputs/data_drift_report.json
        """,
        execution_timeout=timedelta(minutes=8),
        outlets=[MODEL_DATASET],
    )

    retraining_decision = BashOperator(
        task_id="retraining_decision",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python /opt/project/ml/retraining_decision.py

        test -f /opt/project/outputs/retraining_decision.json
        """,
        execution_timeout=timedelta(minutes=8),
        outlets=[MODEL_DATASET],
    )

    check_retraining_gate = BashOperator(
        task_id="check_retraining_gate",
        bash_command="""
        set -euo pipefail

        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python /opt/project/ml/check_retraining_gate.py
        """,
        execution_timeout=timedelta(minutes=5),
    )

    retrain_spend_prediction = BashOperator(
        task_id="retrain_spend_prediction",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/models /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python /opt/project/ml/train_spend_prediction.py

        test -f /opt/project/models/spend_prediction_random_forest.pkl
        test -f /opt/project/outputs/spend_prediction_metrics.json
        test -f /opt/project/outputs/spend_predictions.parquet
        test -f /opt/project/outputs/spend_feature_importance.csv
        test -f /opt/project/outputs/spend_feature_importance.png
        """,
        execution_timeout=timedelta(minutes=12),
        outlets=[MODEL_DATASET],
    )

    model_promotion_decision = BashOperator(
        task_id="model_promotion_decision",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python /opt/project/ml/model_promotion_decision.py

        test -f /opt/project/outputs/model_promotion_decision.json
        """,
        execution_timeout=timedelta(minutes=8),
        outlets=[MODEL_DATASET],
    )

    model_registry_post_retrain = BashOperator(
        task_id="model_registry_post_retrain",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/models/registry /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python /opt/project/ml/model_registry.py

        test -f /opt/project/outputs/model_registry.json
        """,
        execution_timeout=timedelta(minutes=8),
        outlets=[MODEL_DATASET],
    )

    experiment_tracking = BashOperator(
        task_id="experiment_tracking",
        bash_command="""
        set -euo pipefail

        mkdir -p /opt/project/outputs
        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python /opt/project/ml/experiment_tracking.py

        test -f /opt/project/outputs/experiment_tracking.json
        test -f /opt/project/outputs/experiment_history.csv
        """,
        execution_timeout=timedelta(minutes=8),
        outlets=[MODEL_DATASET],
    )

    # ----------------
    # Gold -> Iceberg
    # ----------------
    gold_to_iceberg = BashOperator(
        task_id="gold_to_iceberg",
        bash_command=r"""
        set -euo pipefail

        /opt/spark/bin/spark-sql \
          --master spark://spark-master:7077 \
          --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.10.1 \
          --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
          --conf spark.sql.catalog.lakehouse=org.apache.iceberg.spark.SparkCatalog \
          --conf spark.sql.catalog.lakehouse.type=hive \
          --conf spark.sql.catalog.lakehouse.uri=thrift://hive-metastore:9083 \
          --conf spark.sql.catalog.lakehouse.warehouse=/opt/project/data/iceberg/warehouse \
          --conf spark.sql.shuffle.partitions=4 \
          --conf spark.default.parallelism=4 \
          -f /opt/project/pipelines/create_iceberg_tables.sql
        """,
        execution_timeout=timedelta(minutes=15),
        outlets=[ICEBERG_DATASET],
    )

    # ----------------------------
    # Apache Iceberg
    # ----------------------------
    iceberg_upsert = BashOperator(
        task_id="iceberg_upsert",
        bash_command=r"""
        set -euo pipefail

        /opt/spark/bin/spark-sql \
          --master spark://spark-master:7077 \
          --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.10.1 \
          --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
          --conf spark.sql.catalog.lakehouse=org.apache.iceberg.spark.SparkCatalog \
          --conf spark.sql.catalog.lakehouse.type=hive \
          --conf spark.sql.catalog.lakehouse.uri=thrift://hive-metastore:9083 \
          --conf spark.sql.catalog.lakehouse.warehouse=/opt/project/data/iceberg/warehouse \
          --conf spark.sql.shuffle.partitions=4 \
          --conf spark.default.parallelism=4 \
          -f /opt/project/pipelines/iceberg_upserts.sql
        """,
        execution_timeout=timedelta(minutes=15),
    )

    validate_iceberg = BashOperator(
        task_id="validate_iceberg",
        bash_command=r"""
        set -euo pipefail

        /opt/spark/bin/spark-sql \
          --master spark://spark-master:7077 \
          --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.10.1 \
          --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
          --conf spark.sql.catalog.lakehouse=org.apache.iceberg.spark.SparkCatalog \
          --conf spark.sql.catalog.lakehouse.type=hive \
          --conf spark.sql.catalog.lakehouse.uri=thrift://hive-metastore:9083 \
          --conf spark.sql.catalog.lakehouse.warehouse=/opt/project/data/iceberg/warehouse \
          -e "
            SHOW TABLES IN lakehouse.banking;
            SELECT COUNT(*) AS daily_kpis_rows FROM lakehouse.banking.daily_kpis;
            SELECT COUNT(*) AS customer_features_rows FROM lakehouse.banking.customer_features;
            SELECT COUNT(*) AS merchant_risk_rows FROM lakehouse.banking.merchant_risk;
          "
        """,
        execution_timeout=timedelta(minutes=10),
    )

    iceberg_maintenance = BashOperator(
        task_id="iceberg_maintenance",
        bash_command=r"""
        set -euo pipefail

        /opt/spark/bin/spark-sql \
          --master spark://spark-master:7077 \
          --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.10.1 \
          --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
          --conf spark.sql.catalog.lakehouse=org.apache.iceberg.spark.SparkCatalog \
          --conf spark.sql.catalog.lakehouse.type=hive \
          --conf spark.sql.catalog.lakehouse.uri=thrift://hive-metastore:9083 \
          --conf spark.sql.catalog.lakehouse.warehouse=/opt/project/data/iceberg/warehouse \
          -f /opt/project/pipelines/warehouse/iceberg_maintenance.sql
        """,
        execution_timeout=timedelta(minutes=15),
    )

    # ----------------------------
    # Analytics (DuckDB)
    # ----------------------------
    run_analytics = BashOperator(
        task_id="run_analytics",
        bash_command="""
        set -euo pipefail

        unset PYTHONNOUSERSITE || true
        export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

        /home/airflow/.local/bin/python -c "import duckdb; print('duckdb OK', duckdb.__version__)"
        /home/airflow/.local/bin/python /opt/project/analytics/run_analytics.py
        """,
        execution_timeout=timedelta(minutes=10),
    )
    generate_streaming_bi_summaries = BashOperator(
    task_id="generate_streaming_bi_summaries",
    bash_command="""
    set -euo pipefail

    mkdir -p /opt/project/bi_exports
    unset PYTHONNOUSERSITE || true
    export PYTHONPATH="/home/airflow/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

    /home/airflow/.local/bin/python -c "import pandas, pyarrow; print('Streaming BI summary libs OK')"
    /home/airflow/.local/bin/python /opt/project/analytics/generate_streaming_bi_summaries.py

    test -f /opt/project/bi_exports/transaction_trend_summary.csv
    test -f /opt/project/bi_exports/channel_summary.csv
    test -f /opt/project/bi_exports/geography_summary.csv
    test -f /opt/project/bi_exports/merchant_category_summary.csv
    test -f /opt/project/bi_exports/location_mismatch_summary.csv
    test -f /opt/project/bi_exports/hourly_activity_summary.csv
    """,
    execution_timeout=timedelta(minutes=8),
)

    # ----------------------------
    # Row count validation (Spark)
    # ----------------------------
    validate_row_counts = BashOperator(
        task_id="validate_row_counts",
        bash_command=r"""
        set -euo pipefail
        cd /opt/project

        /opt/spark/bin/spark-shell \
          --master spark://spark-master:7077 \
          --driver-memory 2g \
          --executor-memory 2g \
          -i /dev/stdin <<'SCALA'
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val bronze = spark.read.parquet("data/bronze/transactions")
val silver = spark.read.parquet("data/silver/transactions")

val bronzeCount = bronze.count()
val silverCount = silver.count()

println(s"Bronze count: $bronzeCount")
println(s"Silver count: $silverCount")

if (bronzeCount == 0) throw new RuntimeException("ERROR: Bronze is empty!")
if (silverCount == 0) throw new RuntimeException("ERROR: Silver is empty!")

val ratio = silverCount.toDouble / bronzeCount.toDouble
println(s"Silver/Bronze ratio: $ratio")

spark.stop()
System.exit(0)
SCALA

        python3 - <<'PY'
min_ratio = float("{{ params.min_silver_bronze_ratio }}")
print("Min allowed ratio:", min_ratio)
PY
        """,
        execution_timeout=timedelta(minutes=15),
    )

    # ----------------------------
    # Observability: log metrics
    # ----------------------------
    log_metrics = BashOperator(
        task_id="log_metrics",
        trigger_rule=TriggerRule.ALL_DONE,
        bash_command="""
        set -euo pipefail

        echo "==== DATASET SIZES ===="
        du -sh /opt/project/data/bronze 2>/dev/null || true
        du -sh /opt/project/data/silver 2>/dev/null || true
        du -sh /opt/project/data/gold 2>/dev/null || true
        du -sh /opt/project/data/iceberg 2>/dev/null || true
        du -sh /opt/project/models 2>/dev/null || true
        du -sh /opt/project/outputs 2>/dev/null || true
        ls -lh /opt/project/outputs 2>/dev/null || true

        echo "==== OUTPUT FILES ===="
        ls -lh /opt/project/outputs 2>/dev/null || true

        echo "==== PARQUET FILE COUNTS ===="
        echo -n "bronze parquet: " && find /opt/project/data/bronze -name "*.parquet" 2>/dev/null | wc -l || true
        echo -n "silver parquet: " && find /opt/project/data/silver -name "*.parquet" 2>/dev/null | wc -l || true
        echo -n "gold parquet: " && find /opt/project/data/gold -name "*.parquet" 2>/dev/null | wc -l || true

        echo "==== MODEL FILES ===="
        ls -lh /opt/project/models 2>/dev/null || true
        """,
        execution_timeout=timedelta(minutes=3),
    )

    # ----------------------------
    # Azure layered upload
    # ----------------------------
    azure_layered_upload = BashOperator(
    task_id="azure_layered_upload",
    trigger_rule=TriggerRule.ALL_SUCCESS,
    bash_command="""
    set -euo pipefail

    cd /opt/project/airflow_docker
    python cloud/upload_to_layered_containers.py --upload-iceberg --upload-ml --upload-bi
    """,
    execution_timeout=timedelta(minutes=10),
    )
    
    # Azure SQL Layer
 
        # Azure SQL Layer

    load_bi_exports_to_azure_sql = PythonOperator(
    task_id="load_bi_exports_to_azure_sql",
    python_callable=run_load_blob_to_sql,
    execution_timeout=timedelta(minutes=60),
    )

    # ----------------------------
    # Markers
    # ----------------------------
    write_success_marker = BashOperator(
        task_id="write_success_marker",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        bash_command="""
        set -euo pipefail
        mkdir -p /opt/project/data/_markers
        echo "{{ ds }} {{ run_id }}" > /opt/project/data/_markers/week4_lakehouse_orchestration_success.txt
        """,
        execution_timeout=timedelta(minutes=1),
    )

    write_failure_marker = BashOperator(
        task_id="write_failure_marker",
        trigger_rule=TriggerRule.ONE_FAILED,
        bash_command="""
        set -euo pipefail
        mkdir -p /opt/project/data/_markers
        echo "{{ ds }} {{ run_id }}" > /opt/project/data/_markers/week4_lakehouse_orchestration_failed.txt
        """,
        execution_timeout=timedelta(minutes=1),
    )

    # ----------------------------
    # Flow
    # ----------------------------
    sync_latest_bronze_from_azure_task >> get_latest_stream_batch_file_task >> generate_streaming_bi_summaries >> bronze_to_silver
    bronze_to_silver >> validate_silver >> silver_to_gold >> validate_gold

    validate_gold >> train_customer_segmentation >> segmentation_pca_visualization >> segmentation_feature_profiles >> detect_customer_anomalies >> generate_risk_scores >> enrich_risk_outputs >> train_spend_prediction >> model_monitoring >> model_registry >> data_drift_detection

    data_drift_detection >> run_customer_segmentation_inference
    run_customer_segmentation_inference >> run_spend_prediction_inference >> validate_ml_outputs

    data_drift_detection >> retraining_decision >> check_retraining_gate >> retrain_spend_prediction >> model_registry_post_retrain >> model_promotion_decision >> experiment_tracking

    [validate_ml_outputs, experiment_tracking] >> gold_to_iceberg

    gold_to_iceberg >> iceberg_upsert >> validate_iceberg >> iceberg_maintenance
    iceberg_maintenance >> [validate_row_counts, run_analytics]
    [validate_row_counts, run_analytics] >> log_metrics
    log_metrics >> azure_layered_upload >> load_bi_exports_to_azure_sql >> write_success_marker
    [azure_layered_upload, load_bi_exports_to_azure_sql] >> write_failure_marker
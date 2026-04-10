from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {"owner": "team", "retries": 2, "retry_delay": timedelta(minutes=2)}

with DAG(
    dag_id="week4_lakehouse_orchestration",
    default_args=default_args,
    start_date=datetime(2026, 2, 1),
    schedule=None,
    catchup=False,
    tags=["week4", "lakehouse", "banking"],
) as dag:

    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command=f"cd {REPO_PATH} && source venv/bin/activate && spark-submit pipelines/bronze_to_silver.py",
    )

    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command=f"cd {REPO_PATH} && source venv/bin/activate && spark-submit pipelines/silver_to_gold.py",
    )

    bronze_to_silver >> silver_to_gold

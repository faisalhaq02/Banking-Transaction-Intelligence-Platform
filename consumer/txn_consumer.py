import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kafka import KafkaConsumer

TOPIC = "bank_txn_stream_v2"
BOOTSTRAP = "localhost:9092"

BATCH_SIZE = 500
BATCH_INTERVAL_SECONDS = 300

BASE_DIR = Path(__file__).resolve().parents[1]
TEMP_DIR = BASE_DIR / "data" / "kafka_consumer_batches"
UPLOAD_SCRIPT = BASE_DIR / "airflow_docker" / "cloud" / "azure_blob_stream_upload.py"

TEMP_DIR.mkdir(parents=True, exist_ok=True)

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP,
    group_id="bank_txn_blob_uploader_v2",   # use a fresh group id for replay
    auto_offset_reset="earliest",           # read old backfill data too
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

def write_batch_file(records: list[dict]) -> Path:
    now = datetime.now(timezone.utc)
    file_name = f"transactions_{now.strftime('%Y%m%dT%H%M%S%f')}.jsonl"
    file_path = TEMP_DIR / file_name

    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return file_path

def upload_to_azure(file_path: Path) -> None:
    if not UPLOAD_SCRIPT.exists():
        raise FileNotFoundError(f"Azure upload script not found: {UPLOAD_SCRIPT}")

    result = subprocess.run(
        [sys.executable, str(UPLOAD_SCRIPT), str(file_path)],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR / "airflow_docker"),
    )

    if result.stdout:
        print(result.stdout.strip())

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.strip())
        raise RuntimeError(f"Upload failed for {file_path}")

print("Consuming Kafka messages, batching, and uploading to Azure... Ctrl+C to stop.")

buffer: list[dict] = []
last_flush = time.time()

try:
    for msg in consumer:
        buffer.append(msg.value)

        now = time.time()
        should_flush = (
            len(buffer) >= BATCH_SIZE
            or (now - last_flush) >= BATCH_INTERVAL_SECONDS
        )

        if should_flush and buffer:
            batch_file = write_batch_file(buffer)
            print(f"Created batch file: {batch_file} with {len(buffer)} records")

            upload_to_azure(batch_file)
            print(f"Uploaded batch: {batch_file.name}")

            buffer.clear()
            last_flush = now

except KeyboardInterrupt:
    print("Stopping consumer...")

    if buffer:
        batch_file = write_batch_file(buffer)
        print(f"Created final batch file: {batch_file} with {len(buffer)} records")

        upload_to_azure(batch_file)
        print(f"Uploaded final batch: {batch_file.name}")

finally:
    consumer.close()
    print("Kafka consumer closed.")
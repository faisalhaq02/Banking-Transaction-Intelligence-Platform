import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python azure_blob_stream_upload.py <file_path>")

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("CONTAINER_NAME", "bronze")

    print("DEBUG connection_string exists:", bool(connection_string))
    print("DEBUG container_name:", container_name)

    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")

    local_file = Path(sys.argv[1])
    if not local_file.exists():
        raise FileNotFoundError(f"File not found: {local_file}")

    now = datetime.now(timezone.utc)
    blob_path = (
        f"streaming_transactions/"
        f"year={now.year}/month={now.month:02d}/day={now.day:02d}/hour={now.hour:02d}/"
        f"{local_file.name}"
    )

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_path,
    )

    with local_file.open("rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"Uploaded {local_file} -> {blob_path}")


if __name__ == "__main__":
    main()

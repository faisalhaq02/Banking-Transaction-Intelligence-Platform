from __future__ import annotations

import argparse
import mimetypes
import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Azure container mapping
CONTAINERS = {
    "models": "models",
    "iceberg": "iceberg",
    "bi_exports": "bi-exports",
}

# ML output upload rules
ML_UPLOAD_RULES = [
    {
        "container": "models",
        "base_paths": [
            "/opt/project/models",
            "/opt/project/outputs",
        ],
        "allowed_extensions": {".json", ".csv", ".parquet", ".png", ".txt"},
        "exclude_prefixes": [
            "/opt/project/outputs/bronze",
            "/opt/project/outputs/silver",
            "/opt/project/outputs/gold",
            "/opt/project/outputs/iceberg",
            "/opt/project/outputs/logs",
            "/opt/project/outputs/checkpoints",
            "/opt/project/outputs/tmp",
            "/opt/project/outputs/temp",
        ],
    },
]

# BI export upload rules
BI_EXPORT_UPLOAD_RULES = [
    {
        "container": "bi_exports",
        "base_paths": [
            "/opt/project/bi_exports",
        ],
        "allowed_extensions": {".csv", ".parquet", ".json", ".txt"},
        "exclude_prefixes": [],
    },
]

# Iceberg upload rules
ICEBERG_UPLOAD_RULES = [
    {
        "container": "iceberg",
        "base_paths": [
            "/opt/project/data/iceberg",
        ],
        "allowed_extensions": None,  # upload everything
        "exclude_prefixes": [],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload selected project layers to Azure Blob containers."
    )
    parser.add_argument(
        "--upload-ml",
        action="store_true",
        help="Upload ML models and outputs to the models container",
    )
    parser.add_argument(
        "--upload-iceberg",
        action="store_true",
        help="Upload Iceberg data to the iceberg container",
    )
    parser.add_argument(
        "--upload-bi",
        action="store_true",
        help="Upload BI exports to the bi-exports container",
    )
    return parser.parse_args()


def ensure_connection_string() -> str:
    if not AZURE_CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")
    return AZURE_CONNECTION_STRING


def get_blob_service_client() -> BlobServiceClient:
    conn_str = ensure_connection_string()
    return BlobServiceClient.from_connection_string(conn_str)


def should_exclude(file_path: Path, exclude_prefixes: list[str]) -> bool:
    file_path_str = str(file_path)
    for prefix in exclude_prefixes:
        if file_path_str.startswith(prefix):
            return True
    return False


def get_content_settings(file_path: Path) -> ContentSettings | None:
    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type:
        return ContentSettings(content_type=content_type)
    return None


def upload_file(
    blob_service_client: BlobServiceClient,
    local_file: Path,
    container_name: str,
    blob_name: str,
) -> None:
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
        print(f"Created container: {container_name}")
    except Exception:
        pass

    blob_client = container_client.get_blob_client(blob_name)

    with open(local_file, "rb") as data:
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=get_content_settings(local_file),
        )

    print(f"Uploaded {local_file} -> {container_name}/{blob_name}")


def upload_directory(
    blob_service_client: BlobServiceClient,
    base_path: str,
    container_name: str,
    allowed_extensions: set[str] | None = None,
    exclude_prefixes: list[str] | None = None,
) -> None:
    exclude_prefixes = exclude_prefixes or []
    base = Path(base_path)

    if not base.exists():
        print(f"Skipping missing path: {base}")
        return

    if not base.is_dir():
        print(f"Skipping non-directory path: {base}")
        return

    uploaded_any = False

    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue

        if should_exclude(file_path, exclude_prefixes):
            continue

        if allowed_extensions is not None and file_path.suffix.lower() not in allowed_extensions:
            continue

        # Preserve folder structure relative to /opt/project
        try:
            relative_blob_path = file_path.relative_to("/opt/project")
        except ValueError:
            relative_blob_path = file_path.relative_to(base.parent)

        blob_name = str(relative_blob_path).replace("\\", "/")
        upload_file(
            blob_service_client=blob_service_client,
            local_file=file_path,
            container_name=container_name,
            blob_name=blob_name,
        )
        uploaded_any = True

    if not uploaded_any:
        print(f"No matching files found in {base_path} for container {container_name}")


def run_rules(rules: list[dict]) -> None:
    blob_service_client = get_blob_service_client()

    for rule in rules:
        container_key = rule["container"]
        container_name = CONTAINERS[container_key]
        base_paths = rule["base_paths"]
        allowed_extensions = rule["allowed_extensions"]
        exclude_prefixes = rule["exclude_prefixes"]

        print(f"\nUploading to container: {container_name}")
        for base_path in base_paths:
            upload_directory(
                blob_service_client=blob_service_client,
                base_path=base_path,
                container_name=container_name,
                allowed_extensions=allowed_extensions,
                exclude_prefixes=exclude_prefixes,
            )


def main() -> None:
    args = parse_args()

    # If no flags are provided, upload everything
    no_flags = not args.upload_ml and not args.upload_iceberg and not args.upload_bi
    upload_ml = args.upload_ml or no_flags
    upload_iceberg = args.upload_iceberg or no_flags
    upload_bi = args.upload_bi or no_flags

    print("Starting selected Azure uploads...\n")

    if upload_ml:
        print("Uploading ML outputs...")
        run_rules(ML_UPLOAD_RULES)

    if upload_bi:
        print("\nUploading BI exports...")
        run_rules(BI_EXPORT_UPLOAD_RULES)

    if upload_iceberg:
        print("\nUploading Iceberg data...")
        run_rules(ICEBERG_UPLOAD_RULES)

    print("\nSelected uploads complete.")


if __name__ == "__main__":
    main()
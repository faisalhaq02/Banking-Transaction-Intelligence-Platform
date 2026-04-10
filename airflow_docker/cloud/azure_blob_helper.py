import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


load_dotenv()

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "test-container")


def get_blob_service_client():
    if not CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in .env")
    return BlobServiceClient.from_connection_string(CONNECTION_STRING)


def ensure_container(container_name: str = CONTAINER_NAME):
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)

    if not container_client.exists():
        container_client.create_container()
        print(f"Created container: {container_name}")

    return container_client


def upload_file_to_blob(local_file_path: str, blob_name: str = None, container_name: str = CONTAINER_NAME):
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")

    container_client = ensure_container(container_name)

    if blob_name is None:
        blob_name = os.path.basename(local_file_path)

    blob_client = container_client.get_blob_client(blob_name)

    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"Uploaded '{local_file_path}' to container '{container_name}' as '{blob_name}'")
    return blob_client.url
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_CONTAINER_NAME", "test-container")

if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

if not container_client.exists():
    container_client.create_container()
    print(f"Created container: {container_name}")

SEARCH_DIRS = ["/opt/project/bi_exports", "/opt/project/outputs"]
ALLOWED_EXTENSIONS = {".csv", ".parquet", ".json", ".png"}

for base_dir in SEARCH_DIRS:
    for root, _, files in os.walk(base_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                file_path = os.path.join(root, file_name)
                blob_name = file_path.replace("/opt/project/", "")
                print(f"Uploading: {file_path}")
                with open(file_path, "rb") as data:
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_client.upload_blob(data, overwrite=False)
                print(f"Uploaded as: {blob_name}\n")
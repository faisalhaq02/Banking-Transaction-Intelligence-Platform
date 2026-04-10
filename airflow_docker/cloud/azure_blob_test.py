import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")

if not connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in .env")

if not account_name:
    raise ValueError("AZURE_STORAGE_ACCOUNT_NAME not found in .env")

try:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    containers = blob_service_client.list_containers()

    print("Azure connection successful")
    print(f"Storage account: {account_name}")
    print("Containers:")

    found = False
    for container in containers:
        found = True
        print(f"- {container['name']}")

    if not found:
        print("No containers found in this storage account.")

except Exception as e:
    print("Azure connection failed")
    print(f"Error: {e}")
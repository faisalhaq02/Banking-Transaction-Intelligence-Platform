from azure_blob_helper import upload_file_to_blob

file_url = upload_file_to_blob("test_upload.txt")
print("Uploaded file URL:")
print(file_url)
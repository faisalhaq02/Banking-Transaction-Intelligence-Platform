import os
from azure_blob_helper import upload_file_to_blob

FILES_TO_UPLOAD = [
    "../bi_exports/executive_kpis.csv",
    "../bi_exports/risk_anomaly_summary.csv",
    "../bi_exports/spend_prediction_summary.csv",
    "../outputs/customer_segments.parquet",
    "../outputs/customer_risk_scores.parquet",
    "../outputs/customer_anomalies.parquet",
    "../outputs/anomaly_summary.json",
    "../outputs/retraining_decision.json",
    "../outputs/risk_bucket_summary.json",
]

for file_path in FILES_TO_UPLOAD:
    if os.path.exists(file_path):
        blob_name = file_path.replace("../", "")
        print(f"Uploading: {file_path}")
        url = upload_file_to_blob(file_path, blob_name=blob_name)
        print(f"Uploaded to: {url}\n")
    else:
        print(f"Skipped (not found): {file_path}\n")
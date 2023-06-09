import os
import json
from time import sleep
from prefect_gcp import GcpCredentials, GcsBucket

def create_gcp_creds_block():
    json_file_path = os.getenv("GCP_CREDENTIALS_JSON_FILE_PATH")
    with open(json_file_path) as f:
        service_account_info = json.load(f)

    my_gcp_creds_obj = GcpCredentials(service_account_info=service_account_info)
    my_gcp_creds_obj.save(name="my-gcp-creds", overwrite=True)

def create_gcs_bucket_block():
    gcp_creds = GcpCredentials.load("my-gcp-creds")
    gcp_creds.get_cloud_storage_client().create_bucket("mlops-zoomcamp-bucket-2023ss")

    my_gcs_bucket_obj = GcsBucket(
        bucket="mlops-zoomcamp-bucket-2023ss", gcp_credentials=gcp_creds
    )
    my_gcs_bucket_obj.save(name="mlops-zoomcamp-bucket-2023ss", overwrite=True)

if __name__ == "__main__":
    create_gcp_creds_block()
    sleep(5)
    create_gcs_bucket_block()
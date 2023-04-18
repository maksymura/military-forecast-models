import boto3
import os

bucket_name = "military-forecast"
def get_s3_data(folder_prefix):
    session = boto3.Session(
        aws_access_key_id="AKIATV5NCJMWJHKI44PA",
        aws_secret_access_key="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",
        region_name="eu-central-1"
    )

    s3 = session.client("s3")

    folder_prefix = folder_prefix
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)

    for obj in objects['Contents']:
        file_key = obj['Key']
        local_file_path = os.path.join("resources/" + folder_prefix, os.path.basename(file_key))

        s3.download_file(bucket_name, file_key, local_file_path)
        print(f"Downloaded {file_key} from {bucket_name} to {local_file_path}")

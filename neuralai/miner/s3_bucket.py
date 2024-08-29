import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError

load_dotenv()

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.getenv('AWS_REGION')
BUCKET_NAME = 'neural-ai'

def s3_upload(file_name, object_name=None):
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client('s3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION_NAME
    )

    try:
        s3_client.upload_file(file_name, BUCKET_NAME, object_name)
        print(f"File {file_name} uploaded to {BUCKET_NAME}/{object_name}")
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
    except NoCredentialsError:
        print("Credentials not available.")
    except ClientError as e:
        print(f"Failed to upload {file_name}: {e}")

        
def generate_presigned_url(object_name, expiration=3600):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION_NAME
    )

    try:
        response = s3_client.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': object_name}, ExpiresIn=expiration)
        return response
    except Exception as e:
        print(f"Error generating pre-signed URL: {e}")
        return None
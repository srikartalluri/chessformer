import boto3

# When running locally, boto3 will check environment variables and ~/.aws/credentials.
# On an EC2 instance with an attached IAM role, it will automatically fetch the credentials.
s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket='your-bucket', Prefix='your-prefix/')
for obj in response.get('Contents', []):
    print(obj['Key'])


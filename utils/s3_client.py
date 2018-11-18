#!/usr/bin/env python3

import boto3
import os


class S3Client:

    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ['aws_access_key_id'],
            aws_secret_access_key=os.environ['aws_secret_access_key'],
            region_name='us-east-1')

    def upload(self, local_file_path, s3_bucket_name, s3_file_path):
        """
        Upload the local file to specified s3 bucket

        Param(s):
            local_file_path  The local path of to be uploaded file
            s3_bucket_name   The target s3 bucket's name for storing file
            s3_file_path     The s3 side file path via uploaded endpoint
        """
        self.s3_client.upload_file(local_file_path,
                                   s3_bucket_name,
                                   s3_file_path)


if __name__ == '__main__':
    s3 = S3Client()
    s3.upload('./test.txt', 'test-bucket-aaa-bbb', 'aa/test.txt')

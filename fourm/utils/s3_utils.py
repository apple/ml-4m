# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os
import re
import boto3
import webdataset as wds

from boto3.s3.transfer import TransferConfig
from webdataset.handlers import reraise_exception


def setup_s3_args(args):
    if not args.s3_data_endpoint:
        args.s3_data_endpoint = args.s3_endpoint
    

def save_on_s3(filename, s3_path, s3_endpoint):

    s3_client = boto3.client(
                service_name='s3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                endpoint_url=s3_endpoint
            )

    _, bucket, key, _ = re.split("s3://(.*?)/(.*)$", s3_path)

    s3_client.upload_file(filename, bucket, key)


def download_from_s3(s3_path, s3_endpoint, filename, multipart_threshold_mb=512, multipart_chunksize_mb=512):

    MB = 1024 ** 2
    transfer_config = TransferConfig(
        multipart_threshold=multipart_threshold_mb * MB, 
        multipart_chunksize=multipart_chunksize_mb * MB, 
        max_io_queue=1000)


    s3_client = boto3.client(
                service_name='s3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                endpoint_url=s3_endpoint,
            )

    _, bucket, key, _ = re.split("s3://(.*?)/(.*)$", s3_path)

    s3_client.download_file(bucket, key, filename, Config=transfer_config)



def override_wds_s3_tar_loading(s3_data_endpoint, s3_multipart_threshold_mb, s3_multipart_chunksize_mb, s3_max_io_queue):
    
    # When loading from S3 using boto3, hijack webdatasets tar loading
    MB = 1024 ** 2
    transfer_config = TransferConfig(
        multipart_threshold=s3_multipart_threshold_mb * MB, 
        multipart_chunksize=s3_multipart_chunksize_mb * MB, 
        max_io_queue=s3_max_io_queue)

    s3_client = boto3.client(
        service_name='s3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        endpoint_url=s3_data_endpoint,
    )

    def get_bytes_io(path):
        byte_io = io.BytesIO()
        _, bucket, key, _ = re.split("s3://(.*?)/(.*)$", path)
        s3_client.download_fileobj(bucket, key, byte_io, Config=transfer_config)
        byte_io.seek(0)
        return byte_io
    
    def gopen_with_s3(url, mode="rb", bufsize=8192, **kw):
        """gopen from webdataset, but with s3 support"""
        if url.startswith("s3://"):
            return get_bytes_io(url)
        else:
            return wds.gopen.gopen(url, mode, bufsize, **kw)

    def url_opener(data, handler=reraise_exception, **kw):
        for sample in data:
            url = sample["url"]
            try:
                stream = gopen_with_s3(url, **kw)
                # stream = get_bytes_io(url)
                sample.update(stream=stream)
                yield sample
            except Exception as exn:
                exn.args = exn.args + (url,)
                if handler(exn):
                    continue
                else:
                    break

    wds.tariterators.url_opener = url_opener



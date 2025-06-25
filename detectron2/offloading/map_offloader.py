import os
import json
import boto3
from urllib.parse import urljoin
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MapOffloader:
    """
    Uploads the full map, preview map, and tile images to S3.
    Otherwise retains all fields in the input metadata and transfers them 
    to the output metadata. 

    @param metadata_path: Path to the metadata file.
    @param s3_bucket_name: Name of the S3 bucket to upload to.
    @param region: Region of the S3 bucket.
    @param output_metadata_path: Path to the output metadata file.
    @param expiry_seconds: Expiry time for the presigned URLs.
    """

    def __init__(self,
                 metadata_path,
                 s3_bucket_name="forestfomo",
                 region="ap-south-1",
                 output_metadata_path=None,
                 expiry_seconds=86400):
        self.metadata_path = metadata_path
        self.bucket_name = s3_bucket_name
        self.region = region
        self.expiry_seconds = expiry_seconds
        self.output_metadata_path = output_metadata_path or os.path.join(
            os.path.dirname(metadata_path), "signed_tile_metadata.json")

        self.s3 = boto3.client("s3", region_name=region)
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.s3.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.region}
            )
            self.s3.put_public_access_block(
                Bucket=self.bucket_name,
                PublicAccessBlockConfiguration={
                    'BlockPublicAcls': True,
                    'IgnorePublicAcls': True,
                    'BlockPublicPolicy': True,
                    'RestrictPublicBuckets': True
                }
            )

    def _upload_and_sign(self, local_path, s3_key):
        self.s3.upload_file(local_path, self.bucket_name, s3_key)
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': s3_key},
            ExpiresIn=self.expiry_seconds
        )

    def process(self):
        """Uploads the full map, preview map, and tile images to S3.

        Otherwise retains all fields in the input metadata and transfers them 
        to the output metadata. 
        """
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        site_groups = defaultdict(list)
        for entry in metadata:
            site_name = entry['parent']['name']
            site_groups[site_name].append(entry)

        for site_name, entries in site_groups.items():
            logger.info(f"Processing site: {site_name}")
            # Upload full + preview maps
            full_path = entries[0]['parent']['image']['source']
            full_key = f"maps/{site_name}/full/{os.path.basename(full_path)}"
            full_url = self._upload_and_sign(full_path, full_key)

            preview_path = entries[0]['parent']['image'].get('preview')
            if preview_path:
                preview_key = f"maps/{site_name}/previews/{os.path.basename(preview_path)}"
                preview_url = self._upload_and_sign(preview_path, preview_key)
            else:
                preview_url = None

            # Upload tile images
            len_entries = len(entries)
            for i, entry in enumerate(entries):
                logger.info(
                    f"Uploading tile {i+1}/{len_entries} for site: {site_name}")
                tile_path = entry['image']['source']
                tile_key = f"maps/{site_name}/tiles/{os.path.basename(tile_path)}"
                tile_url = self._upload_and_sign(tile_path, tile_key)

                entry['image']['source'] = tile_url
                entry['parent']['image']['source'] = full_url
                entry['parent']['image']['preview'] = preview_url

        with open(self.output_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        root_output_key = "maps/signed_tile_metadata.json"
        self.s3.upload_file(self.output_metadata_path,
                            self.bucket_name, root_output_key)

        logger.info(
            f"Signed metadata written to {self.output_metadata_path} and uploaded to S3 at {root_output_key}.")

        return self.output_metadata_path

# data/s3_data_loader.py

import logging
from pathlib import Path
from typing import Optional
from io import BytesIO

import boto3
from botocore.exceptions import ClientError
from PIL import Image

logger = logging.getLogger(__name__)

class S3DataLoader:
    """
    Utility class to load images from AWS S3.
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        cache_dir: str = "cache/s3_images"
    ):
        """
        Initializes the S3DataLoader with AWS credentials and S3 bucket details.
        
        Args:
            bucket_name (str): Name of the S3 bucket.
            prefix (str): Prefix/path within the S3 bucket to look for images.
            region_name (str, optional): AWS region name.
            aws_access_key_id (str, optional): AWS access key ID.
            aws_secret_access_key (str, optional): AWS secret access key.
            cache_dir (str): Local directory to cache downloaded images.
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client = boto3.client(
                's3',
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            logger.info(f"S3 client initialized for bucket '{self.bucket_name}' with prefix '{self.prefix}'.")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def _get_local_path(self, key: str) -> Path:
        """
        Generates the local cache path for a given S3 key.
        
        Args:
            key (str): S3 object key.
        
        Returns:
            Path: Local file path where the image will be cached.
        """
        relative_path = Path(key).relative_to(self.prefix) if self.prefix else Path(key)
        local_path = self.cache_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return local_path

    def load_image(self, key: str, mode: str = "RGB") -> Image.Image:
        """
        Loads an image from S3, using a local cache to minimize S3 requests.
        
        Args:
            key (str): S3 object key for the image.
            mode (str): Mode to convert the image. Defaults to "RGB".
        
        Returns:
            Image.Image: PIL Image object.
        
        Raises:
            FileNotFoundError: If the image cannot be found in S3.
            IOError: If the image cannot be opened or is corrupted.
        """
        local_path = self._get_local_path(key)

        if local_path.is_file():
            try:
                with Image.open(local_path) as img:
                    img = img.convert(mode)
                logger.debug(f"Loaded image from cache: '{local_path}'.")
                return img
            except Exception as e:
                logger.warning(f"Failed to load cached image '{local_path}': {e}. Attempting to re-download.")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = response['Body'].read()
            img = Image.open(BytesIO(image_data)).convert(mode)
            img.save(local_path)
            logger.info(f"Downloaded and cached image '{key}' to '{local_path}'.")
            return img
        except ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                logger.error(f"Image '{key}' does not exist in bucket '{self.bucket_name}'.")
                raise FileNotFoundError(f"Image '{key}' not found in bucket '{self.bucket_name}'.") from e
            else:
                logger.error(f"Failed to download image '{key}' from S3: {e}")
                raise
        except Exception as e:
            logger.error(f"Error processing image '{key}': {e}")
            raise

    def list_images(self, extension: Optional[List[str]] = None) -> List[str]:
        """
        Lists all image keys in the specified S3 bucket and prefix.
        
        Args:
            extension (List[str], optional): List of file extensions to filter images. E.g., ['.jpg', '.png']
        
        Returns:
            List[str]: List of image keys.
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

        image_keys = []
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if extension:
                        if any(key.lower().endswith(ext.lower()) for ext in extension):
                            image_keys.append(key)
                    else:
                        image_keys.append(key)
        logger.info(f"Found {len(image_keys)} images in bucket '{self.bucket_name}' with prefix '{self.prefix}'.")
        return image_keys

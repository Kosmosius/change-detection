# data/s3_data_loader.py

import logging
from pathlib import Path
from typing import Optional, List, Union
from io import BytesIO
import threading

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
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
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initializes the S3DataLoader with AWS S3 bucket details.

        Args:
            bucket_name (str): Name of the S3 bucket.
            prefix (str): Prefix/path within the S3 bucket to look for images.
            region_name (Optional[str]): AWS region name.
            cache_dir (Optional[Union[str, Path]]): Local directory to cache downloaded images.

        Raises:
            ValueError: If required parameters are missing.
            NoCredentialsError: If AWS credentials are not found.
            PartialCredentialsError: If incomplete AWS credentials are provided.
            ClientError: If there's an error connecting to S3.
        """
        if not bucket_name:
            logger.error("Bucket name must be provided.")
            raise ValueError("Bucket name must be provided.")

        self.bucket_name = bucket_name
        self.prefix = prefix
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/s3_images")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # For thread-safe file access

        try:
            # Use default AWS credentials if not provided
            self.s3_client = boto3.client('s3', region_name=region_name)
            logger.info("S3 client initialized for bucket '%s' with prefix '%s'.", self.bucket_name, self.prefix)
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("AWS credentials not found or incomplete: %s", e)
            raise
        except ClientError as e:
            logger.error("Failed to initialize S3 client: %s", e)
            raise

    def _get_local_path(self, key: str) -> Path:
        """
        Generates the local cache path for a given S3 key.

        Args:
            key (str): S3 object key.

        Returns:
            Path: Local file path where the image will be cached.
        """
        if self.prefix and key.startswith(self.prefix):
            relative_key = key[len(self.prefix):].lstrip('/')
        else:
            relative_key = key
        local_path = self.cache_dir / relative_key
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
            ClientError: If there's an error communicating with S3.
        """
        local_path = self._get_local_path(key)

        # Attempt to load from cache
        if local_path.is_file():
            try:
                with local_path.open('rb') as f:
                    with Image.open(f) as img:
                        img = img.convert(mode)
                logger.debug("Loaded image from cache: '%s'.", local_path)
                return img
            except (IOError, OSError) as e:
                logger.warning("Failed to load cached image '%s': %s. Re-downloading.", local_path, e)

        # Download from S3 with thread-safe file access
        with self._lock:
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                image_data = response['Body'].read()
                img = Image.open(BytesIO(image_data)).convert(mode)

                # Save to cache
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with local_path.open('wb') as f:
                    img.save(f)
                logger.info("Downloaded and cached image '%s' to '%s'.", key, local_path)
                return img
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == "NoSuchKey":
                    logger.error("Image '%s' does not exist in bucket '%s'.", key, self.bucket_name)
                    raise FileNotFoundError(f"Image '{key}' not found in bucket '{self.bucket_name}'.") from e
                else:
                    logger.error("Failed to download image '%s' from S3: %s", key, e)
                    raise
            except (IOError, OSError) as e:
                logger.error("Error processing image '%s': %s", key, e)
                raise

    def list_images(self, extensions: Optional[List[str]] = None) -> List[str]:
        """
        Lists all image keys in the specified S3 bucket and prefix.

        Args:
            extensions (Optional[List[str]]): List of file extensions to filter images. E.g., ['.jpg', '.png']

        Returns:
            List[str]: List of image keys.

        Raises:
            ClientError: If there's an error communicating with S3.
        """
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix)

        image_keys = []
        try:
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if extensions:
                            if any(key.lower().endswith(ext.lower()) for ext in extensions):
                                image_keys.append(key)
                        else:
                            image_keys.append(key)
            logger.info("Found %d images in bucket '%s' with prefix '%s'.", len(image_keys), self.bucket_name, self.prefix)
            return image_keys
        except ClientError as e:
            logger.error("Failed to list images in bucket '%s': %s", self.bucket_name, e)
            raise

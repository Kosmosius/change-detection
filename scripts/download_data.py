# scripts/download_data.py

import logging
import os
import sys
import time
from typing import List, Tuple, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.config_parser import get_config
from utils.logger import setup_logger


def create_local_directory(file_path: str) -> None:
    """
    Create the local directory for the given file path if it does not exist.

    Parameters
    ----------
    file_path : str
        The full path to the file for which the directory needs to be created.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug("Created directory: %s", directory)


def download_file_from_s3(s3_client, bucket: str, s3_key: str, local_path: str,
                          retry_attempts: int, retry_delay: float) -> None:
    """
    Download a single file from S3 to a local path with retry logic.

    Parameters
    ----------
    s3_client : boto3.client
        The S3 client object.
    bucket : str
        The name of the S3 bucket.
    s3_key : str
        The key of the S3 object to download.
    local_path : str
        The local file path where the object will be saved.
    retry_attempts : int
        Number of retry attempts in case of failure.
    retry_delay : float
        Delay in seconds between retry attempts.

    Raises
    ------
    ClientError
        If the download fails after all retry attempts.
    """
    for attempt in range(1, retry_attempts + 1):
        try:
            logger.debug("Attempting to download %s to %s (Attempt %d)", s3_key, local_path, attempt)
            s3_client.download_file(bucket, s3_key, local_path)
            logger.info("Successfully downloaded %s to %s", s3_key, local_path)
            return
        except (ClientError, EndpointConnectionError, NoCredentialsError) as e:
            logger.warning("Failed to download %s (Attempt %d/%d): %s", s3_key, attempt, retry_attempts, e)
            if attempt < retry_attempts:
                time.sleep(retry_delay)
            else:
                logger.error("Exceeded maximum retry attempts for %s", s3_key)
                raise e


def map_s3_key(s3_prefix: Optional[str], local_path: str) -> str:
    """
    Map a local file path to its corresponding S3 key based on the provided S3 prefix.

    Parameters
    ----------
    s3_prefix : Optional[str]
        The prefix in the S3 bucket where the data is stored.
    local_path : str
        The local file path.

    Returns
    -------
    str
        The corresponding S3 key.
    """
    if s3_prefix:
        # Remove leading '/' if present in local_path to avoid issues with S3 keys
        relative_path = os.path.relpath(local_path, start=os.path.commonpath([local_path]))
        s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
    else:
        s3_key = local_path.replace("\\", "/")
    return s3_key


def gather_data_paths(config: dict) -> List[Tuple[str, str]]:
    """
    Gather all local and corresponding S3 paths for images and labels.

    Parameters
    ----------
    config : dict
        The training configuration dictionary.

    Returns
    -------
    List[Tuple[str, str]]
        A list of tuples where each tuple contains (local_path, s3_key).
    """
    data_paths = []

    # Collect train image pairs
    for pair in config['data']['train_image_pairs']:
        for local_path in pair:
            s3_key = map_s3_key(config['data']['s3_prefix'], local_path)
            data_paths.append((local_path, s3_key))

    # Collect train labels
    for local_path in config['data']['train_labels']:
        s3_key = map_s3_key(config['data']['s3_prefix'], local_path)
        data_paths.append((local_path, s3_key))

    # Collect validation image pairs
    for pair in config['data']['val_image_pairs']:
        for local_path in pair:
            s3_key = map_s3_key(config['data']['s3_prefix'], local_path)
            data_paths.append((local_path, s3_key))

    # Collect validation labels
    for local_path in config['data']['val_labels']:
        s3_key = map_s3_key(config['data']['s3_prefix'], local_path)
        data_paths.append((local_path, s3_key))

    return data_paths


def download_data(config: dict, s3_client) -> None:
    """
    Download all specified data from S3 to local paths.

    Parameters
    ----------
    config : dict
        The training configuration dictionary.
    s3_client : boto3.client
        The S3 client object.
    """
    data_paths = gather_data_paths(config)
    total_files = len(data_paths)
    logger.info("Starting download of %d files from S3.", total_files)

    with ThreadPoolExecutor(max_workers=config['data']['num_workers']) as executor:
        future_to_file = {}
        for local_path, s3_key in data_paths:
            create_local_directory(local_path)
            future = executor.submit(
                download_file_from_s3,
                s3_client,
                config['data']['s3_bucket'],
                s3_key,
                local_path,
                config['data']['retry_attempts'],
                config['data']['retry_delay']
            )
            future_to_file[future] = (s3_key, local_path)

        for future in as_completed(future_to_file):
            s3_key, local_path = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                logger.error("Failed to download %s to %s: %s", s3_key, local_path, e)


def validate_local_data(config: dict) -> None:
    """
    Validate that all local data paths exist.

    Parameters
    ----------
    config : dict
        The training configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If any of the specified local data files do not exist.
    """
    missing_files = []

    # Check train image pairs
    for pair in config['data']['train_image_pairs']:
        for local_path in pair:
            if not os.path.isfile(local_path):
                missing_files.append(local_path)

    # Check train labels
    for local_path in config['data']['train_labels']:
        if not os.path.isfile(local_path):
            missing_files.append(local_path)

    # Check validation image pairs
    for pair in config['data']['val_image_pairs']:
        for local_path in pair:
            if not os.path.isfile(local_path):
                missing_files.append(local_path)

    # Check validation labels
    for local_path in config['data']['val_labels']:
        if not os.path.isfile(local_path):
            missing_files.append(local_path)

    if missing_files:
        logger.error("The following data files are missing:")
        for file in missing_files:
            logger.error("- %s", file)
        raise FileNotFoundError("Some data files are missing. Please ensure all files are downloaded.")


def main() -> None:
    """
    Main function to orchestrate the data downloading process.
    """
    # Setup logger
    logger = setup_logger(__name__, log_dir="logs", log_file="download_data.log", level=logging.INFO)
    logger.info("Logger initialized successfully.")

    # Parse configuration
    config = get_config()
    logger.info("Configuration parsed successfully.")

    if config['data']['use_s3']:
        if not config['data']['s3_bucket']:
            logger.error("S3 bucket name must be provided when 'use_s3' is set to true.")
            sys.exit(1)
        # Initialize S3 client
        try:
            s3_client = boto3.client('s3')
            logger.info("Initialized S3 client successfully.")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure your AWS credentials.")
            sys.exit(1)
        except Exception as e:
            logger.error("Failed to initialize S3 client: %s", e)
            sys.exit(1)

        # Download data from S3
        try:
            download_data(config, s3_client)
            logger.info("Data downloaded successfully from S3.")
        except Exception as e:
            logger.error("Data download failed: %s", e)
            sys.exit(1)
    else:
        logger.info("S3 usage is disabled. Skipping data download from S3.")

    # Validate local data
    try:
        validate_local_data(config)
        logger.info("All local data files are present.")
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    logger.info("Data preparation completed successfully.")


if __name__ == "__main__":
    main()

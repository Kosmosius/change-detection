# scripts/preprocess_data.py

import logging
import os
import sys
import time
from typing import List, Tuple, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, UnidentifiedImageError

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


def resize_image(input_path: str, output_path: str, size: Tuple[int, int]) -> None:
    """
    Resize an image to the specified size and save it to the output path.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Path to save the resized image.
    size : Tuple[int, int]
        Desired size as (width, height).

    Raises
    ------
    ValueError
        If the image cannot be opened or processed.
    """
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")  # Ensure image is in RGB format
            img_resized = img.resize(size, Image.ANTIALIAS)
            create_local_directory(output_path)
            img_resized.save(output_path)
            logger.info("Resized image saved to %s", output_path)
    except UnidentifiedImageError:
        logger.error("Cannot identify image file %s", input_path)
        raise ValueError(f"Cannot identify image file {input_path}")
    except Exception as e:
        logger.error("Error processing image %s: %s", input_path, e)
        raise ValueError(f"Error processing image {input_path}: {e}")


def normalize_image(input_path: str, output_path: str, mean: Tuple[float, float, float],
                   std: Tuple[float, float, float]) -> None:
    """
    Normalize an image and save it to the output path.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Path to save the normalized image.
    mean : Tuple[float, float, float]
        Mean values for each channel.
    std : Tuple[float, float, float]
        Standard deviation values for each channel.

    Raises
    ------
    ValueError
        If the image cannot be opened or processed.
    """
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img_normalized = img.copy()
            pixels = img_normalized.load()

            for i in range(img_normalized.width):
                for j in range(img_normalized.height):
                    r, g, b = pixels[i, j]
                    r = (r / 255.0 - mean[0]) / std[0]
                    g = (g / 255.0 - mean[1]) / std[1]
                    b = (b / 255.0 - mean[2]) / std[2]
                    pixels[i, j] = (int(r * 255), int(g * 255), int(b * 255))

            create_local_directory(output_path)
            img_normalized.save(output_path)
            logger.info("Normalized image saved to %s", output_path)
    except UnidentifiedImageError:
        logger.error("Cannot identify image file %s", input_path)
        raise ValueError(f"Cannot identify image file {input_path}")
    except Exception as e:
        logger.error("Error normalizing image %s: %s", input_path, e)
        raise ValueError(f"Error normalizing image {input_path}: {e}")


def preprocess_image(
    input_path: str,
    output_path: str,
    size: Tuple[int, int],
    normalize: bool,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float]
) -> None:
    """
    Preprocess a single image by resizing and optionally normalizing it.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_path : str
        Path to save the preprocessed image.
    size : Tuple[int, int]
        Desired size as (width, height).
    normalize : bool
        Whether to normalize the image.
    mean : Tuple[float, float, float]
        Mean values for normalization.
    std : Tuple[float, float, float]
        Standard deviation values for normalization.
    """
    try:
        resize_image(input_path, output_path, size)
        if normalize:
            normalize_image(output_path, output_path, mean, std)
    except ValueError as e:
        logger.error("Preprocessing failed for %s: %s", input_path, e)


def gather_image_paths(config: dict) -> List[Tuple[str, str]]:
    """
    Gather all input and output image paths for preprocessing.

    Parameters
    ----------
    config : dict
        The training configuration dictionary.

    Returns
    -------
    List[Tuple[str, str]]
        A list of tuples where each tuple contains (input_path, output_path).
    """
    image_pairs = config['data']['train_image_pairs'] + config['data']['val_image_pairs']
    labels = config['data']['train_labels'] + config['data']['val_labels']
    data_paths = []

    # Process image pairs
    for pair in image_pairs:
        before_img, after_img = pair
        output_before = before_img.replace("raw", "processed")
        output_after = after_img.replace("raw", "processed")
        data_paths.append((before_img, output_before))
        data_paths.append((after_img, output_after))

    # Process labels
    for label in labels:
        output_label = label.replace("raw", "processed")
        data_paths.append((label, output_label))

    return data_paths


def validate_image_paths(data_paths: List[Tuple[str, str]]) -> None:
    """
    Validate that all input image paths exist.

    Parameters
    ----------
    data_paths : List[Tuple[str, str]]
        List of tuples containing (input_path, output_path).

    Raises
    ------
    FileNotFoundError
        If any input image file does not exist.
    """
    missing_files = [input_path for input_path, _ in data_paths if not os.path.isfile(input_path)]
    if missing_files:
        logger.error("The following input image files are missing:")
        for file in missing_files:
            logger.error("- %s", file)
        raise FileNotFoundError("Some input image files are missing. Please ensure all files are available.")


def preprocess_data(config: dict) -> None:
    """
    Preprocess all images by resizing and normalizing as specified in the configuration.

    Parameters
    ----------
    config : dict
        The training configuration dictionary.
    """
    data_paths = gather_image_paths(config)
    total_files = len(data_paths)
    logger.info("Starting preprocessing of %d files.", total_files)

    with ThreadPoolExecutor(max_workers=config['preprocessing']['num_workers']) as executor:
        future_to_file = {}
        for input_path, output_path in data_paths:
            future = executor.submit(
                preprocess_image,
                input_path,
                output_path,
                tuple(config['preprocessing']['resize']),
                config['preprocessing']['normalize'],
                tuple(config['preprocessing']['mean']),
                tuple(config['preprocessing']['std'])
            )
            future_to_file[future] = input_path

        for future in as_completed(future_to_file):
            input_path = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                logger.error("Preprocessing failed for %s: %s", input_path, e)


def main() -> None:
    """
    Main function to orchestrate the data preprocessing process.
    """
    # Setup logger
    logger = setup_logger(__name__, log_dir="logs", log_file="preprocess_data.log", level=logging.INFO)
    logger.info("Logger initialized successfully.")

    # Parse configuration
    config = get_config()
    logger.info("Configuration parsed successfully.")

    # Validate preprocessing configuration
    preprocessing_config = config.get('preprocessing', {})
    required_keys = ['resize', 'normalize', 'mean', 'std', 'num_workers']
    for key in required_keys:
        if key not in preprocessing_config:
            logger.error("Missing preprocessing configuration key: '%s'", key)
            sys.exit(1)

    # Validate image paths
    data_paths = gather_image_paths(config)
    try:
        validate_image_paths(data_paths)
        logger.info("All input image files are present.")
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # Preprocess data
    try:
        preprocess_data(config)
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error("Data preprocessing failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

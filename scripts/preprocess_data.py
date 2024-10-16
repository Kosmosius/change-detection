# scripts/preprocess_data.py

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, UnidentifiedImageError

from utils.config_parser import get_config
from utils.logger import setup_logger

# Initialize the logger at the module level
logger = logging.getLogger(__name__)


def create_local_directory(file_path: Path) -> None:
    """
    Create the local directory for the given file path if it does not exist.

    Parameters
    ----------
    file_path : Path
        The full path to the file for which the directory needs to be created.
    """
    directory = file_path.parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory: %s", directory)


def resize_image(input_path: Path, output_path: Path, size: Tuple[int, int]) -> None:
    """
    Resize an image to the specified size and save it to the output path.

    Parameters
    ----------
    input_path : Path
        Path to the input image.
    output_path : Path
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
            img_resized = img.resize(size, Image.LANCZOS)
            create_local_directory(output_path)
            img_resized.save(output_path)
            logger.info("Resized image saved to %s", output_path)
    except UnidentifiedImageError:
        logger.error("Cannot identify image file %s", input_path)
        raise ValueError(f"Cannot identify image file {input_path}")
    except Exception as e:
        logger.error("Error processing image %s: %s", input_path, e)
        raise ValueError(f"Error processing image {input_path}: {e}")


def normalize_image(
    input_path: Path,
    output_path: Path,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> None:
    """
    Normalize an image and save it to the output path.

    Parameters
    ----------
    input_path : Path
        Path to the input image.
    output_path : Path
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
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = (img_array - mean) / std
            img_array = np.clip(img_array, 0.0, 1.0)
            img_array = (img_array * 255.0).astype(np.uint8)
            img_normalized = Image.fromarray(img_array)
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
    input_path: Path,
    output_path: Path,
    size: Tuple[int, int],
    normalize: bool,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> None:
    """
    Preprocess a single image by resizing and optionally normalizing it.

    Parameters
    ----------
    input_path : Path
        Path to the input image.
    output_path : Path
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


def gather_image_paths(config: dict) -> List[Tuple[Path, Path]]:
    """
    Gather all input and output image paths for preprocessing.

    Parameters
    ----------
    config : dict
        The training configuration dictionary.

    Returns
    -------
    List[Tuple[Path, Path]]
        A list of tuples where each tuple contains (input_path, output_path).
    """
    data_config = config.get("data", {})
    input_base_dir = Path(data_config.get("input_base_dir", "data/raw")).resolve()
    output_base_dir = Path(data_config.get("output_base_dir", "data/processed")).resolve()
    image_pairs = data_config.get("train_image_pairs", []) + data_config.get("val_image_pairs", [])
    labels = data_config.get("train_labels", []) + data_config.get("val_labels", [])
    data_paths = []

    # Process image pairs
    for pair in image_pairs:
        for img_path_str in pair:
            input_path = input_base_dir / img_path_str
            relative_path = Path(img_path_str)
            output_path = output_base_dir / relative_path
            data_paths.append((input_path, output_path))

    # Process labels
    for label_path_str in labels:
        input_path = input_base_dir / label_path_str
        relative_path = Path(label_path_str)
        output_path = output_base_dir / relative_path
        data_paths.append((input_path, output_path))

    return data_paths


def validate_image_paths(data_paths: List[Tuple[Path, Path]]) -> None:
    """
    Validate that all input image paths exist.

    Parameters
    ----------
    data_paths : List[Tuple[Path, Path]]
        List of tuples containing (input_path, output_path).

    Raises
    ------
    FileNotFoundError
        If any input image file does not exist.
    """
    missing_files = [input_path for input_path, _ in data_paths if not input_path.is_file()]
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

    preprocessing_config = config.get("preprocessing", {})
    size = tuple(preprocessing_config.get("resize", (256, 256)))
    normalize = preprocessing_config.get("normalize", True)
    mean = tuple(preprocessing_config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(preprocessing_config.get("std", (0.229, 0.224, 0.225)))
    num_workers = preprocessing_config.get("num_workers", 4)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {}
        for input_path, output_path in data_paths:
            future = executor.submit(
                preprocess_image,
                input_path,
                output_path,
                size,
                normalize,
                mean,
                std,
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
    setup_logger(__name__, log_dir="logs", log_file="preprocess_data.log", level=logging.INFO)
    logger.info("Logger initialized successfully.")

    # Parse configuration
    config = get_config()
    logger.info("Configuration parsed successfully.")

    # Validate preprocessing configuration
    preprocessing_config = config.get("preprocessing", {})
    required_keys = ["resize", "normalize", "mean", "std", "num_workers"]
    missing_keys = [key for key in required_keys if key not in preprocessing_config]
    if missing_keys:
        logger.error("Missing preprocessing configuration keys: %s", ", ".join(missing_keys))
        sys.exit(1)

    # Gather and validate image paths
    try:
        data_paths = gather_image_paths(config)
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

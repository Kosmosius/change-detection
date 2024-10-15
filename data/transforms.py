# data/transforms.py

import logging
from typing import Callable, Optional, Dict, Any, List, Union
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ToTensor,
    Normalize,
)

logger = logging.getLogger(__name__)


def get_resize_transform(resize: List[int]) -> Resize:
    """
    Returns a Resize transformation.

    Args:
        resize (List[int]): Desired output size as [height, width].

    Returns:
        Resize: The Resize transformation.

    Raises:
        ValueError: If resize parameters are invalid.
    """
    if not isinstance(resize, (list, tuple)) or len(resize) != 2:
        logger.error("Invalid resize parameters: %s", resize)
        raise ValueError(f"Invalid resize parameters: {resize}")
    return Resize(resize)


def get_random_horizontal_flip_transform(prob: float = 0.5) -> RandomHorizontalFlip:
    """
    Returns a RandomHorizontalFlip transformation.

    Args:
        prob (float): Probability of flipping the image.

    Returns:
        RandomHorizontalFlip: The RandomHorizontalFlip transformation.

    Raises:
        ValueError: If probability is not between 0 and 1.
    """
    if not (0.0 <= prob <= 1.0):
        logger.error("Invalid probability for RandomHorizontalFlip: %f", prob)
        raise ValueError(f"Invalid probability for RandomHorizontalFlip: {prob}")
    return RandomHorizontalFlip(prob)


def get_random_vertical_flip_transform(prob: float = 0.5) -> RandomVerticalFlip:
    """
    Returns a RandomVerticalFlip transformation.

    Args:
        prob (float): Probability of flipping the image.

    Returns:
        RandomVerticalFlip: The RandomVerticalFlip transformation.

    Raises:
        ValueError: If probability is not between 0 and 1.
    """
    if not (0.0 <= prob <= 1.0):
        logger.error("Invalid probability for RandomVerticalFlip: %f", prob)
        raise ValueError(f"Invalid probability for RandomVerticalFlip: {prob}")
    return RandomVerticalFlip(prob)


def get_random_rotation_transform(degrees: float = 15.0) -> RandomRotation:
    """
    Returns a RandomRotation transformation.

    Args:
        degrees (float): Range of degrees for random rotations.

    Returns:
        RandomRotation: The RandomRotation transformation.

    Raises:
        ValueError: If degrees is negative.
    """
    if degrees < 0:
        logger.error("Invalid degrees for RandomRotation: %f", degrees)
        raise ValueError(f"Invalid degrees for RandomRotation: {degrees}")
    return RandomRotation(degrees)


def get_to_tensor_transform() -> ToTensor:
    """
    Returns a ToTensor transformation.

    Returns:
        ToTensor: The ToTensor transformation.
    """
    return ToTensor()


def get_normalize_transform(mean: List[float], std: List[float]) -> Normalize:
    """
    Returns a Normalize transformation.

    Args:
        mean (List[float]): Mean values for each channel.
        std (List[float]): Standard deviation values for each channel.

    Returns:
        Normalize: The Normalize transformation.

    Raises:
        ValueError: If mean and std lists are of different lengths.
    """
    if len(mean) != len(std):
        logger.error("Mean and std must be of the same length. Mean: %s, Std: %s", mean, std)
        raise ValueError("Mean and std must be of the same length.")
    return Normalize(mean=mean, std=std)


def get_composed_transform(transform_config: Dict[str, Any]) -> Compose:
    """
    Constructs a composed transformation pipeline based on the provided configuration.

    Args:
        transform_config (Dict[str, Any]): Configuration dictionary for transformations.

    Returns:
        Compose: The composed transformation pipeline.

    Raises:
        ValueError: If the configuration is invalid.
    """
    transform_list: List[Callable] = []

    # Resize
    resize_config = transform_config.get('resize')
    if resize_config:
        transform_list.append(get_resize_transform(resize_config))
        logger.debug("Added Resize transform with size %s", resize_config)

    # Random Horizontal Flip
    hflip_config = transform_config.get('random_horizontal_flip')
    if hflip_config:
        prob = hflip_config.get('probability', 0.5)
        transform_list.append(get_random_horizontal_flip_transform(prob))
        logger.debug("Added RandomHorizontalFlip transform with probability %f", prob)

    # Random Vertical Flip
    vflip_config = transform_config.get('random_vertical_flip')
    if vflip_config:
        prob = vflip_config.get('probability', 0.5)
        transform_list.append(get_random_vertical_flip_transform(prob))
        logger.debug("Added RandomVerticalFlip transform with probability %f", prob)

    # Random Rotation
    rotation_config = transform_config.get('random_rotation')
    if rotation_config:
        degrees = rotation_config.get('degrees', 15.0)
        transform_list.append(get_random_rotation_transform(degrees))
        logger.debug("Added RandomRotation transform with degrees %f", degrees)

    # ToTensor
    transform_list.append(get_to_tensor_transform())
    logger.debug("Added ToTensor transform")

    # Normalize
    normalize_config = transform_config.get('normalize')
    if normalize_config:
        mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
        std = normalize_config.get('std', [0.229, 0.224, 0.225])
        transform_list.append(get_normalize_transform(mean, std))
        logger.debug("Added Normalize transform with mean %s and std %s", mean, std)

    composed_transform = Compose(transform_list)
    logger.info("Composed transformation pipeline created successfully.")
    return composed_transform


def get_default_transforms() -> Compose:
    """
    Returns the default set of transformations to apply to the images.

    Returns:
        Compose: Composed transformations.
    """
    default_config = {
        'resize': [256, 256],
        'random_horizontal_flip': {'probability': 0.5},
        'random_vertical_flip': {'probability': 0.5},
        'random_rotation': {'degrees': 15},
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
    }
    return get_composed_transform(default_config)


def get_custom_transforms(transform_config: Optional[Dict[str, Any]] = None) -> Compose:
    """
    Returns a custom set of transformations based on the provided configuration.

    Args:
        transform_config (Optional[Dict[str, Any]]): Custom configuration for transformations.
            If None, returns the default transformations.

    Returns:
        Compose: Composed transformations.

    Raises:
        ValueError: If the custom configuration is invalid.
    """
    if transform_config is None:
        logger.info("No custom transform configuration provided. Using default transforms.")
        return get_default_transforms()

    return get_composed_transform(transform_config)


def load_transforms_from_config(config_path: Union[str, Path]) -> Compose:
    """
    Loads transformation configurations from a YAML file and constructs the transformation pipeline.

    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file.

    Returns:
        Compose: Composed transformations.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration is invalid.
    """
    from omegaconf import OmegaConf

    config_path = Path(config_path)
    if not config_path.is_file():
        logger.error("Transform configuration file '%s' does not exist.", config_path)
        raise FileNotFoundError(f"Transform configuration file '{config_path}' does not exist.")

    try:
        config = OmegaConf.load(config_path)
        transform_config = OmegaConf.to_container(config, resolve=True)
        composed_transform = get_custom_transforms(transform_config)
        logger.info("Transformations loaded from configuration file '%s'.", config_path)
        return composed_transform
    except (OmegaConf.BaseContainer, ValueError) as e:
        logger.error("Failed to load transformations from '%s': %s", config_path, e)
        raise ValueError(f"Invalid transform configuration in '{config_path}'.") from e

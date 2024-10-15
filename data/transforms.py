# data/transforms.py

import logging
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToTensor, Normalize
from PIL import Image

logger = logging.getLogger(__name__)


class TransformationError(Exception):
    """Custom exception for transformation-related errors."""
    pass


def get_resize_transform(resize: List[int]) -> Resize:
    """
    Returns a Resize transformation.
    
    Args:
        resize (List[int]): Desired output size as [height, width].
    
    Returns:
        Resize: The Resize transformation.
    """
    try:
        return Resize(resize)
    except Exception as e:
        logger.error(f"Failed to create Resize transform with size {resize}: {e}")
        raise TransformationError(f"Invalid resize parameters: {resize}") from e


def get_random_horizontal_flip_transform(prob: float = 0.5) -> RandomHorizontalFlip:
    """
    Returns a RandomHorizontalFlip transformation.
    
    Args:
        prob (float): Probability of flipping the image.
    
    Returns:
        RandomHorizontalFlip: The RandomHorizontalFlip transformation.
    """
    try:
        return RandomHorizontalFlip(prob)
    except Exception as e:
        logger.error(f"Failed to create RandomHorizontalFlip transform with probability {prob}: {e}")
        raise TransformationError(f"Invalid probability for RandomHorizontalFlip: {prob}") from e


def get_random_vertical_flip_transform(prob: float = 0.5) -> RandomVerticalFlip:
    """
    Returns a RandomVerticalFlip transformation.
    
    Args:
        prob (float): Probability of flipping the image.
    
    Returns:
        RandomVerticalFlip: The RandomVerticalFlip transformation.
    """
    try:
        return RandomVerticalFlip(prob)
    except Exception as e:
        logger.error(f"Failed to create RandomVerticalFlip transform with probability {prob}: {e}")
        raise TransformationError(f"Invalid probability for RandomVerticalFlip: {prob}") from e


def get_random_rotation_transform(degrees: float = 15.0) -> RandomRotation:
    """
    Returns a RandomRotation transformation.
    
    Args:
        degrees (float): Range of degrees for random rotations.
    
    Returns:
        RandomRotation: The RandomRotation transformation.
    """
    try:
        return RandomRotation(degrees)
    except Exception as e:
        logger.error(f"Failed to create RandomRotation transform with degrees {degrees}: {e}")
        raise TransformationError(f"Invalid degrees for RandomRotation: {degrees}") from e


def get_to_tensor_transform() -> ToTensor:
    """
    Returns a ToTensor transformation.
    
    Returns:
        ToTensor: The ToTensor transformation.
    """
    try:
        return ToTensor()
    except Exception as e:
        logger.error(f"Failed to create ToTensor transform: {e}")
        raise TransformationError("Invalid parameters for ToTensor") from e


def get_normalize_transform(mean: List[float], std: List[float]) -> Normalize:
    """
    Returns a Normalize transformation.
    
    Args:
        mean (List[float]): Mean values for each channel.
        std (List[float]): Standard deviation values for each channel.
    
    Returns:
        Normalize: The Normalize transformation.
    """
    try:
        return Normalize(mean=mean, std=std)
    except Exception as e:
        logger.error(f"Failed to create Normalize transform with mean {mean} and std {std}: {e}")
        raise TransformationError(f"Invalid parameters for Normalize: mean={mean}, std={std}") from e


def get_composed_transform(transform_config: Dict[str, Any]) -> Compose:
    """
    Constructs a composed transformation pipeline based on the provided configuration.
    
    Args:
        transform_config (Dict[str, Any]): Configuration dictionary for transformations.
    
    Returns:
        Compose: The composed transformation pipeline.
    
    Raises:
        TransformationError: If the configuration is invalid.
    """
    try:
        transform_list: List[Callable] = []
        
        # Resize
        resize_config = transform_config.get('resize')
        if resize_config:
            transform_list.append(get_resize_transform(resize_config))
            logger.debug(f"Added Resize transform with size {resize_config}")
        
        # Random Horizontal Flip
        hflip_config = transform_config.get('random_horizontal_flip')
        if hflip_config:
            prob = hflip_config.get('probability', 0.5)
            transform_list.append(get_random_horizontal_flip_transform(prob))
            logger.debug(f"Added RandomHorizontalFlip transform with probability {prob}")
        
        # Random Vertical Flip
        vflip_config = transform_config.get('random_vertical_flip')
        if vflip_config:
            prob = vflip_config.get('probability', 0.5)
            transform_list.append(get_random_vertical_flip_transform(prob))
            logger.debug(f"Added RandomVerticalFlip transform with probability {prob}")
        
        # Random Rotation
        rotation_config = transform_config.get('random_rotation')
        if rotation_config:
            degrees = rotation_config.get('degrees', 15.0)
            transform_list.append(get_random_rotation_transform(degrees))
            logger.debug(f"Added RandomRotation transform with degrees {degrees}")
        
        # ToTensor
        transform_list.append(get_to_tensor_transform())
        logger.debug("Added ToTensor transform")
        
        # Normalize
        normalize_config = transform_config.get('normalize')
        if normalize_config:
            mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
            std = normalize_config.get('std', [0.229, 0.224, 0.225])
            transform_list.append(get_normalize_transform(mean, std))
            logger.debug(f"Added Normalize transform with mean {mean} and std {std}")
        
        composed_transform = Compose(transform_list)
        logger.info("Composed transformation pipeline created successfully.")
        return composed_transform
    except Exception as e:
        logger.error(f"Failed to create composed transform: {e}")
        raise TransformationError("Invalid transformation configuration.") from e


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
            'std': [0.229, 0.224, 0.225]
        }
    }
    return get_composed_transform(default_config)


def get_custom_transforms(transform_config: Optional[Dict[str, Any]] = None) -> Compose:
    """
    Returns a custom set of transformations based on the provided configuration.
    
    Args:
        transform_config (Dict[str, Any], optional): Custom configuration for transformations.
            If None, returns the default transformations.
    
    Returns:
        Compose: Composed transformations.
    
    Raises:
        TransformationError: If the custom configuration is invalid.
    """
    if transform_config is None:
        logger.info("No custom transform configuration provided. Using default transforms.")
        return get_default_transforms()
    
    try:
        custom_transform = get_composed_transform(transform_config)
        logger.info("Custom transformation pipeline created successfully.")
        return custom_transform
    except TransformationError as e:
        logger.error(f"Failed to create custom transforms: {e}")
        raise


# Example of loading transforms from a YAML configuration file
def load_transforms_from_config(config_path: Union[str, Path]) -> Compose:
    """
    Loads transformation configurations from a YAML file and constructs the transformation pipeline.
    
    Args:
        config_path (str or Path): Path to the YAML configuration file.
    
    Returns:
        Compose: Composed transformations.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        TransformationError: If the configuration is invalid.
    """
    from omegaconf import OmegaConf
    
    config_path = Path(config_path)
    if not config_path.is_file():
        logger.error(f"Transform configuration file '{config_path}' does not exist.")
        raise FileNotFoundError(f"Transform configuration file '{config_path}' does not exist.")
    
    try:
        config = OmegaConf.load(config_path)
        transform_config = OmegaConf.to_container(config, resolve=True)
        composed_transform = get_custom_transforms(transform_config)
        logger.info(f"Transformations loaded from configuration file '{config_path}'.")
        return composed_transform
    except Exception as e:
        logger.error(f"Failed to load transformations from '{config_path}': {e}")
        raise TransformationError(f"Invalid transform configuration in '{config_path}'.") from e

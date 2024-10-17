# data/datasets.py

import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict

import logging

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageFile

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .s3_data_loader import S3DataLoader

# To handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class ChangeDetectionDataset(Dataset):
    """
    Custom Dataset for Change Detection tasks.

    Each sample consists of a pair of images (before and after) and a corresponding label.
    Supports loading images from local storage or AWS S3 with caching.
    """

    def __init__(
        self,
        image_pairs: List[Tuple[str, str]],
        labels: List[str],
        transform: Optional[A.Compose] = None,
        use_s3: bool = False,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        cache_transforms: bool = False,
        transform_cache_dir: Optional[Union[str, Path]] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initializes the dataset with image paths and labels.

        Args:
            image_pairs (List[Tuple[str, str]]): List of tuples containing paths to before and after images.
            labels (List[str]): List of paths to label images.
            transform (Optional[A.Compose]): Transformations to apply to the images.
            use_s3 (bool): Whether to load images from AWS S3.
            s3_bucket (Optional[str]): S3 bucket name.
            s3_prefix (Optional[str]): S3 prefix/path.
            cache_transforms (bool): Whether to cache transformed images for faster loading.
            transform_cache_dir (Optional[Union[str, Path]]): Directory to cache transformed images.
            retry_attempts (int): Number of retry attempts for loading images.
            retry_delay (float): Delay between retry attempts in seconds.

        Raises:
            ValueError: If input lists are of unequal length or required parameters are missing.
        """
        if len(image_pairs) != len(labels):
            logger.error("Number of image pairs must match number of labels.")
            raise ValueError("Number of image pairs must match number of labels.")

        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform
        self.use_s3 = use_s3
        self.cache_transforms = cache_transforms
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        if self.use_s3:
            if not s3_bucket or not s3_prefix:
                logger.error("S3 bucket and prefix must be provided when use_s3 is True.")
                raise ValueError("S3 bucket and prefix must be provided when use_s3 is True.")
            self.s3_loader = S3DataLoader(bucket_name=s3_bucket, prefix=s3_prefix)
            logger.info("Initialized S3DataLoader for dataset.")

        # Initialize transformation caching if enabled
        if self.cache_transforms and self.transform:
            if not transform_cache_dir:
                logger.error("transform_cache_dir must be provided when cache_transforms is True.")
                raise ValueError("transform_cache_dir must be provided when cache_transforms is True.")
            self.transform_cache_dir = Path(transform_cache_dir)
            self.transform_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Transformation caching enabled. Cache directory: %s", self.transform_cache_dir)

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves the image pair and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]:
                - 'x_before': Tensor of the 'before' image with shape [C, H, W].
                - 'x_after': Tensor of the 'after' image with shape [C, H, W].
                - 'labels': Tensor of the label image with shape [1, H, W].

        Raises:
            FileNotFoundError: If image files are not found.
            IOError: If images cannot be opened or are corrupted.
        """
        before_path, after_path = self.image_pairs[idx]
        label_path = self.labels[idx]

        try:
            # Load images with retry mechanism
            before_img = self._load_image(before_path, mode="RGB")
            after_img = self._load_image(after_path, mode="RGB")
            label_img = self._load_image(label_path, mode="L")  # Assuming label is grayscale
        except (FileNotFoundError, IOError) as e:
            logger.error("Error loading images for index %d: %s", idx, e)
            raise

        # Convert images to numpy arrays
        before_img = np.array(before_img)
        after_img = np.array(after_img)
        label_img = np.array(label_img)

        # Apply transformations with optional caching
        if self.cache_transforms and self.transform:
            cache_path = self.transform_cache_dir / f"{idx}.pt"
            if cache_path.is_file():
                try:
                    data_dict = torch.load(cache_path)
                    logger.debug("Loaded transformed images from cache for index %d.", idx)
                    return data_dict
                except (OSError, IOError) as e:
                    logger.warning("Failed to load cached transforms for index %d: %s. Re-transforming.", idx, e)

        # Apply the same transformations to images and labels
        if self.transform:
            transformed = self.transform(image=before_img, image0=after_img, mask=label_img)
            before_tensor = transformed['image']    # Tensor of shape [C, H, W]
            after_tensor = transformed['image0']    # Tensor of shape [C, H, W]
            label_tensor = transformed['mask']      # Tensor of shape [H, W]

        else:
            # If no transformations are applied, convert images to tensors
            before_tensor = torch.from_numpy(before_img).permute(2, 0, 1).float() / 255.0  # [C, H, W]
            after_tensor = torch.from_numpy(after_img).permute(2, 0, 1).float() / 255.0    # [C, H, W]
            label_tensor = torch.from_numpy(label_img).unsqueeze(0).float() / 255.0         # [1, H, W]

        # Ensure label has shape [1, H, W] and type float32
        if label_tensor.dim() == 2:
            label_tensor = label_tensor.unsqueeze(0).float()
        elif label_tensor.dim() == 3 and label_tensor.shape[0] == 1:
            label_tensor = label_tensor.float()
        else:
            logger.warning("Unexpected label shape: %s", label_tensor.shape)
            label_tensor = label_tensor.unsqueeze(0).float()

        # If caching is enabled, save the transformed tensors
        if self.cache_transforms and self.transform:
            try:
                data_dict = {
                    'x_before': before_tensor,
                    'x_after': after_tensor,
                    'labels': label_tensor
                }
                torch.save(data_dict, cache_path)
                logger.debug("Saved transformed images to cache for index %d.", idx)
            except (OSError, IOError) as e:
                logger.warning("Failed to save transformed images to cache for index %d: %s.", idx, e)

        return {
            'x_before': before_tensor,
            'x_after': after_tensor,
            'labels': label_tensor
        }

    def _load_image(self, path: str, mode: str = "RGB") -> Image.Image:
        """
        Loads an image from local storage or S3 with retry logic.

        Args:
            path (str): Path to the image.
            mode (str): Mode to convert the image. Defaults to "RGB".

        Returns:
            Image.Image: PIL Image object.

        Raises:
            FileNotFoundError: If the image cannot be found.
            IOError: If the image cannot be opened or is corrupted.
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                if self.use_s3:
                    img = self.s3_loader.load_image(path, mode=mode)
                else:
                    img = Image.open(path).convert(mode)
                return img
            except (FileNotFoundError, IOError) as e:
                logger.warning("Attempt %d failed to load image '%s': %s", attempt, path, e)
                if attempt < self.retry_attempts:
                    logger.info("Retrying in %.2f seconds...", self.retry_delay)
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All %d attempts failed to load image '%s'.", self.retry_attempts, path)
                    raise

def get_default_transforms() -> A.Compose:
    """
    Returns the default transformations for the dataset using albumentations.

    The transformations are applied to both images and labels, ensuring consistency.
    """
    return A.Compose([
        A.Resize(height=224, width=224),  # Resize to match ViT's expected input size
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'image0': 'image', 'mask': 'mask'})

def get_dataloader(
    image_pairs: List[Tuple[str, str]],
    labels: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    transform: Optional[A.Compose] = None,
    use_s3: bool = False,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    num_workers: int = 0,  # Set to 0 for compatibility on Windows
    cache_transforms: bool = False,
    transform_cache_dir: Optional[Union[str, Path]] = None,
    retry_attempts: int = 3,
    retry_delay: float = 1.0
) -> DataLoader:
    """
    Creates a DataLoader for the ChangeDetectionDataset.

    Args:
        image_pairs (List[Tuple[str, str]]): List of image pair paths.
        labels (List[str]): List of label paths.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        transform (Optional[A.Compose], optional): Transformations to apply. Defaults to None.
        use_s3 (bool, optional): Whether to load images from AWS S3. Defaults to False.
        s3_bucket (Optional[str], optional): S3 bucket name. Defaults to None.
        s3_prefix (Optional[str], optional): S3 prefix/path. Defaults to None.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.
        cache_transforms (bool, optional): Whether to cache transformed images. Defaults to False.
        transform_cache_dir (Optional[Union[str, Path]], optional): Directory to cache transformed images. Required if cache_transforms is True.
        retry_attempts (int, optional): Number of retry attempts for loading images. Defaults to 3.
        retry_delay (float, optional): Delay between retry attempts in seconds. Defaults to 1.0.

    Returns:
        DataLoader: Configured DataLoader.

    Raises:
        ValueError: If required parameters are missing.
    """
    if transform is None:
        transform = get_default_transforms()

    dataset = ChangeDetectionDataset(
        image_pairs=image_pairs,
        labels=labels,
        transform=transform,
        use_s3=use_s3,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        cache_transforms=cache_transforms,
        transform_cache_dir=transform_cache_dir,
        retry_attempts=retry_attempts,
        retry_delay=retry_delay
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(
        "DataLoader created with batch_size=%d, shuffle=%s, num_workers=%d.",
        batch_size,
        shuffle,
        num_workers
    )
    return dataloader

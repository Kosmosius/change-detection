# data/datasets.py

import os
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data.s3_data_loader import S3DataLoader
import logging
from functools import partial

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
        transform: Optional[transforms.Compose] = None,
        use_s3: bool = False,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        cache_transforms: bool = False,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initializes the dataset with image paths and labels.

        Args:
            image_pairs (List[Tuple[str, str]]): List of tuples containing paths to before and after images.
            labels (List[str]): List of paths to label images.
            transform (transforms.Compose, optional): Transformations to apply to the images.
            use_s3 (bool): Whether to load images from AWS S3.
            s3_bucket (str, optional): S3 bucket name.
            s3_prefix (str, optional): S3 prefix/path.
            cache_transforms (bool): Whether to cache transformed images for faster loading.
            retry_attempts (int): Number of retry attempts for loading images.
            retry_delay (float): Delay between retry attempts in seconds.
        """
        assert len(image_pairs) == len(labels), "Number of image pairs must match number of labels."
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform
        self.use_s3 = use_s3
        self.cache_transforms = cache_transforms
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        if self.use_s3:
            assert s3_bucket is not None and s3_prefix is not None, "S3 bucket and prefix must be provided when use_s3 is True."
            self.s3_loader = S3DataLoader(bucket_name=s3_bucket, prefix=s3_prefix)
            logger.info("Initialized S3DataLoader for dataset.")

        # Initialize transformation caching if enabled
        if self.cache_transforms and self.transform:
            self.transform_cache_dir = Path("cache/transformed_images")
            self.transform_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Transformation caching enabled. Cache directory: {self.transform_cache_dir}")

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Retrieves the image pair and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
                - Tuple of before and after images as tensors.
                - Label image as a tensor.
        """
        before_path, after_path = self.image_pairs[idx]
        label_path = self.labels[idx]

        try:
            # Load images with retry mechanism
            before_img = self._load_image(before_path, mode="RGB")
            after_img = self._load_image(after_path, mode="RGB")
            label_img = self._load_image(label_path, mode="L")  # Assuming label is grayscale
        except Exception as e:
            logger.error(f"Error loading images for index {idx}: {e}")
            raise

        # Apply transformations with optional caching
        if self.transform:
            before_img, after_img, label_img = self._apply_transforms(idx, before_img, after_img, label_img)

        return (before_img, after_img), label_img

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
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed to load image '{path}': {e}")
                if attempt < self.retry_attempts:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.retry_attempts} attempts failed to load image '{path}'.")
                    raise

    def _apply_transforms(
        self,
        idx: int,
        before_img: Image.Image,
        after_img: Image.Image,
        label_img: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies transformations to the images with optional caching.

        Args:
            idx (int): Index of the sample.
            before_img (Image.Image): Before image.
            after_img (Image.Image): After image.
            label_img (Image.Image): Label image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Transformed before, after, and label images.
        """
        if self.cache_transforms:
            cache_path = self.transform_cache_dir / f"{idx}.pt"
            if cache_path.is_file():
                try:
                    before_tensor, after_tensor, label_tensor = torch.load(cache_path)
                    logger.debug(f"Loaded transformed images from cache for index {idx}.")
                    return before_tensor, after_tensor, label_tensor
                except Exception as e:
                    logger.warning(f"Failed to load cached transforms for index {idx}: {e}. Re-transforming.")

        before_tensor = self.transform(before_img)
        after_tensor = self.transform(after_img)
        label_tensor = transforms.ToTensor()(label_img)  # Convert label to tensor without normalization

        if self.cache_transforms:
            try:
                torch.save((before_tensor, after_tensor, label_tensor), cache_path)
                logger.debug(f"Saved transformed images to cache for index {idx}.")
            except Exception as e:
                logger.warning(f"Failed to save transformed images to cache for index {idx}: {e}.")

        return before_tensor, after_tensor, label_tensor


def get_default_transforms(
    resize: Tuple[int, int] = (256, 256),
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotation_degree: int = 15,
    normalize: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> transforms.Compose:
    """
    Returns the default set of transformations to apply to the images.

    Args:
        resize (Tuple[int, int], optional): Resize dimensions. Defaults to (256, 256).
        horizontal_flip (bool, optional): Whether to apply horizontal flip. Defaults to True.
        vertical_flip (bool, optional): Whether to apply vertical flip. Defaults to True.
        rotation_degree (int, optional): Maximum degree for random rotations. Defaults to 15.
        normalize (bool, optional): Whether to apply normalization. Defaults to True.
        mean (List[float], optional): Mean values for normalization. Defaults to ImageNet means.
        std (List[float], optional): Standard deviation for normalization. Defaults to ImageNet std.

    Returns:
        transforms.Compose: Composed transformations.
    """
    transform_list = [
        transforms.Resize(resize),
    ]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())
    if rotation_degree > 0:
        transform_list.append(transforms.RandomRotation(rotation_degree))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_dataloader(
    image_pairs: List[Tuple[str, str]],
    labels: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    transform: Optional[transforms.Compose] = None,
    use_s3: bool = False,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    num_workers: int = 4,
    cache_transforms: bool = False,
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
        transform (transforms.Compose, optional): Transformations to apply. Defaults to None.
        use_s3 (bool, optional): Whether to load images from AWS S3. Defaults to False.
        s3_bucket (str, optional): S3 bucket name. Defaults to None.
        s3_prefix (str, optional): S3 prefix/path. Defaults to None.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
        cache_transforms (bool, optional): Whether to cache transformed images. Defaults to False.
        retry_attempts (int, optional): Number of retry attempts for loading images. Defaults to 3.
        retry_delay (float, optional): Delay between retry attempts in seconds. Defaults to 1.0.

    Returns:
        DataLoader: Configured DataLoader.
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

    logger.info(f"DataLoader created with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}.")
    return dataloader

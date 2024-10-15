# data/datasets.py

import os
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.s3_data_loader import S3DataLoader
import logging

logger = logging.getLogger(__name__)

class ChangeDetectionDataset(Dataset):
    """
    Custom Dataset for Change Detection tasks.
    
    Each sample consists of a pair of images (before and after) and a corresponding label.
    """

    def __init__(
        self,
        image_pairs: List[Tuple[str, str]],
        labels: List[str],
        transform: Optional[transforms.Compose] = None,
        use_s3: bool = False,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None
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
        """
        assert len(image_pairs) == len(labels), "Number of image pairs must match number of labels."
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform
        self.use_s3 = use_s3

        if self.use_s3:
            assert s3_bucket is not None and s3_prefix is not None, "S3 bucket and prefix must be provided when use_s3 is True."
            self.s3_loader = S3DataLoader(bucket_name=s3_bucket, prefix=s3_prefix)
            logger.info("Initialized S3DataLoader for dataset.")

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
            if self.use_s3:
                before_img = self.s3_loader.load_image(before_path)
                after_img = self.s3_loader.load_image(after_path)
                label_img = self.s3_loader.load_image(label_path, mode='L')  # Assuming label is grayscale
            else:
                before_img = Image.open(before_path).convert("RGB")
                after_img = Image.open(after_path).convert("RGB")
                label_img = Image.open(label_path).convert("L")  # Assuming label is grayscale
        except Exception as e:
            logger.error(f"Error loading images for index {idx}: {e}")
            raise

        if self.transform:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)
            label_img = transforms.ToTensor()(label_img)  # Convert label to tensor without normalization

        return (before_img, after_img), label_img

def get_default_transforms() -> transforms.Compose:
    """
    Returns the default set of transformations to apply to the images.
    
    Returns:
        transforms.Compose: Composed transformations.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet means
                             std=[0.229, 0.224, 0.225])
    ])

def get_dataloader(
    image_pairs: List[Tuple[str, str]],
    labels: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    transform: Optional[transforms.Compose] = None,
    use_s3: bool = False,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for the ChangeDetectionDataset.
    
    Args:
        image_pairs (List[Tuple[str, str]]): List of image pair paths.
        labels (List[str]): List of label paths.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        transform (transforms.Compose, optional): Transformations to apply.
        use_s3 (bool): Whether to load images from AWS S3.
        s3_bucket (str, optional): S3 bucket name.
        s3_prefix (str, optional): S3 prefix/path.
        num_workers (int): Number of subprocesses for data loading.
    
    Returns:
        torch.utils.data.DataLoader: Configured DataLoader.
    """
    if transform is None:
        transform = get_default_transforms()
    
    dataset = ChangeDetectionDataset(
        image_pairs=image_pairs,
        labels=labels,
        transform=transform,
        use_s3=use_s3,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

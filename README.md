# Change Detection

**Author**: [Kosmosius](https://github.com/Kosmosius)

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Models Supported](#models-supported)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

The **Change Detection Project** is an open-source initiative aimed at detecting changes between pairs of images using deep learning techniques. This project focuses on semantic change detection, where the goal is to identify and localize areas of change between two images captured at different times.

This repository provides:

- Implementation of state-of-the-art models for change detection.
- Flexible training pipelines with customizable configurations.
- Support for various loss functions and evaluation metrics.
- Easy integration with datasets and data loaders.
- Options to use custom trainers or leverage HuggingFace's Trainer.

---

## Features

- **Modular Architecture**: Easily extend or modify components like models, loss functions, and metrics.
- **Configurable Training**: Use YAML configuration files to set up experiments without changing code.
- **Multiple Models**: Support for models like Siamese U-Net and Change Detection Transformer.
- **Custom and Predefined Loss Functions**: Including Dice Loss, IoU Loss, Focal Loss, and Binary Cross-Entropy.
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score, IoU, and Dice Coefficient.
- **GPU Support**: Leverage CUDA for faster training on supported hardware.
- **Data Augmentation**: Integrate custom data transformations easily.
- **Checkpointing**: Save and load model checkpoints to resume training or for inference.
- **Logging**: Detailed logging of training progress and metrics.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/) 1.7 or higher
- [Torchvision](https://pytorch.org/vision/stable/)
- Other dependencies listed in `requirements.txt`

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Kosmosius/change-detection.git
   cd change-detection


2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt

4. **Install PyTorch**

Install PyTorch according to your system and CUDA availability. Visit [PyTorch](https://pytorch.org/) for instructions.

---


## Data Preparation
### Dataset Structure
Organize your dataset as follows:

```
Copy code
data/
├── train/
│   ├── images_before/
│   ├── images_after/
│   └── labels/
└── val/
    ├── images_before/
    ├── images_after/
    └── labels/
```

- **images_before/:** Contains images captured at time T1.
- **images_after/:** Contains images captured at time T2.
- **labels/:** Ground truth change masks.

### Supported Formats
- **Images:** PNG, JPEG, or other formats supported by `PIL`.
- **Labels:** Binary masks where changed areas are marked (e.g., pixel value 1 for change, 0 for no change).

### Updating `config.yaml`
Update the paths in `config/config.yaml` to point to your dataset files.

---

## Configuration
The project uses a YAML configuration file located at `config/config.yaml`. This file defines all parameters for model architecture, training, data loading, and more.

### Customization
- **Model Selection:** Choose between `'siamese_unet'` and `'change_detection_transformer'`.
- **Loss Functions:** Options include `'bce'`, `'dice'`, `'iou'`, and `'focal'`.
- **Metrics:** Add or remove metrics as needed.
- **Training Parameters:** Adjust `learning_rate`, `batch_size`, `num_epochs`, etc.
- **Data Paths:** Update `train_image_pairs`, `train_labels`, `val_image_pairs`, and `val_labels` with your dataset paths.

---

## Usage
### Training the Model
1. **Update Configuration**

Ensure `config/config.yaml` is properly set up with your desired parameters and data paths.

2. **Run the Training Script**

   ```bash
   python train/train.py

This will start the training process using the settings from the configuration file.

### Monitoring Training
- **Logs:** Training logs are saved in the logs/ directory.
- **Checkpoints:** Model checkpoints are saved in the checkpoints/ directory after each epoch.

### Resuming Training
To resume training from a checkpoint, update the checkpoint_path in config.yaml:

```yaml
training:
  checkpoint_path: 'checkpoints/checkpoint_epoch_10.pth'
```

---

## Models Supported
### Siamese U-Net
A U-Net based architecture that processes "before" and "after" images separately through shared weights and then combines the features to detect changes.

- **Parameters:**
  - `in_channels`: Number of input channels (e.g., 3 for RGB images).
  - `out_channels`: Number of output channels.
  - `feature_maps`: List defining the number of feature maps at each level.

### Change Detection Transformer
A transformer-based model that leverages attention mechanisms to capture complex changes between images.

- **Parameters:**
  - `encoder_name`: Name of the backbone encoder (e.g., 'resnet50').
  - `num_classes`: Number of output classes.
  - `use_peft`: Whether to use Parameter-Efficient Fine-Tuning.
 
---

## Training
### Custom Trainer
The default training loop is handled by `CustomTrainer`, which provides flexibility and control over the training process.

### HuggingFace Trainer
Optionally, you can use HuggingFace's `Trainer` by setting `use_huggingface_trainer` to `true` in the configuration.

---

## Evaluation
After training, you can evaluate the model using the validation set. Metrics specified in the configuration will be computed and logged.

---

## Results
Results will vary depending on the dataset and parameters used. Monitor the logs for training and validation metrics to assess model performance.

---

## Contributing
Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository**

   ```bash
   git clone https://github.com/Kosmosius/change-detection.git

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name

3. **Make Changes and Commit**

   ```bash
   git commit -am 'Add new feature'

4. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name

5. **Submit a Pull Request**

Open a pull request on the main repository with a clear description of your changes.

---

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit).

---

## Contact
For questions or support, please open an issue on the [GitHub repository](https://github.com/Kosmosius/change-detection).

---

Thank you for using the Change Detection Project!

Feel free to contribute, suggest improvements, or share your results.

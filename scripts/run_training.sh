#!/bin/bash

# scripts/run_training.sh

###############################################################################
# Script to Run Model Training
#
# This script orchestrates the training process of a machine learning model.
# It ensures that the necessary data is downloaded and preprocessed before
# initiating the training. The training process, including logs and checkpoints,
# are managed and recorded for analysis and future reference.
#
# Usage:
#   ./run_training.sh [--config CONFIG_PATH] [--checkpoint CHECKPOINT_PATH]
#
# Options:
#   --config CONFIG_PATH         Path to the configuration YAML file (default: config/config.yaml)
#   --checkpoint CHECKPOINT_PATH Path to the model checkpoint file to resume training (optional)
#
# Example:
#   ./run_training.sh --config config/config.yaml --checkpoint checkpoints/checkpoint_epoch_10.pth
#
###############################################################################

set -euo pipefail

# Default configurations
CONFIG_PATH="config/config.yaml"
CHECKPOINT_PATH=""  # Empty by default; optional

# Function to display usage information
usage() {
    echo "Usage: $0 [--config CONFIG_PATH] [--checkpoint CHECKPOINT_PATH]"
    echo ""
    echo "Options:"
    echo "  --config CONFIG_PATH         Path to the configuration YAML file (default: config/config.yaml)"
    echo "  --checkpoint CHECKPOINT_PATH Path to the model checkpoint file to resume training (optional)"
    echo ""
    echo "Example:"
    echo "  $0 --config config/config.yaml --checkpoint checkpoints/checkpoint_epoch_10.pth"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to check if a file exists
check_file_exists() {
    local file_path="$1"
    if [[ ! -f "$file_path" ]]; then
        echo "Error: File '$file_path' does not exist."
        exit 1
    fi
}

# Function to set up logger
setup_logger() {
    local log_dir="$1"
    local log_file="$2"
    local log_level="$3"

    mkdir -p "$log_dir"

    # Export PYTHONLOGDIR so that Python scripts can use it
    export PYTHONLOGDIR="$log_dir"

    # Create a Python logging configuration file
    cat > "$log_dir/logging.conf" <<EOL
[loggers]
keys=root

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=$log_level
handlers=consoleHandler,fileHandler

[handler_consoleHandler]
class=StreamHandler
level=$log_level
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=$log_level
formatter=simpleFormatter
args=("$log_dir/$log_file", 'a')

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
EOL
}

# Setup logger
setup_logger "logs" "run_training.log" "INFO"
echo "Logger initialized successfully."

# Check if configuration file exists
check_file_exists "$CONFIG_PATH"
echo "Configuration file '$CONFIG_PATH' found."

# Check if checkpoint file exists (if provided)
if [[ -n "$CHECKPOINT_PATH" ]]; then
    check_file_exists "$CHECKPOINT_PATH"
    echo "Checkpoint file '$CHECKPOINT_PATH' found."
else
    echo "No checkpoint file provided. Training will start from scratch."
fi

# Optional: Activate virtual environment
# Uncomment and modify the following lines if using a virtual environment
# VENV_PATH="venv/bin/activate"
# if [[ -f "$VENV_PATH" ]]; then
#     source "$VENV_PATH"
#     echo "Activated virtual environment."
# else
#     echo "Warning: Virtual environment at '$VENV_PATH' not found. Proceeding without activation."
# fi

# Optional: Download data from S3
# Uncomment the following lines if data needs to be downloaded before preprocessing
# echo "Starting data download..."
# python scripts/download_data.py --config "$CONFIG_PATH"
# echo "Data download completed."

# Optional: Preprocess data
# Uncomment the following lines if data needs to be preprocessed before training
# echo "Starting data preprocessing..."
# python scripts/preprocess_data.py --config "$CONFIG_PATH"
# echo "Data preprocessing completed."

# Run the training script
echo "Starting model training..."

# Check if train.py exists
TRAIN_SCRIPT="train/train.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Error: Training script '$TRAIN_SCRIPT' not found."
    echo "Please ensure that 'train/train.py' exists and is executable."
    exit 1
fi

# Execute the training script with the provided configuration and checkpoint
if [[ -n "$CHECKPOINT_PATH" ]]; then
    python "$TRAIN_SCRIPT" --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT_PATH" | tee "logs/training_output.log"
else
    python "$TRAIN_SCRIPT" --config "$CONFIG_PATH" | tee "logs/training_output.log"
fi

echo "Model training completed successfully."

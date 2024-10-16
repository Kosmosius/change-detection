#!/bin/bash

# scripts/run_training.sh

###############################################################################
# Script to Run Model Training
#
# This script orchestrates the training process of a machine learning model.
# It ensures that the necessary data is available before initiating the training.
# The training process, including logs and checkpoints, are managed and recorded
# for analysis and future reference.
#
# Usage:
#   ./run_training.sh [-c CONFIG_PATH] [-k CHECKPOINT_PATH] [-h]
#
# Options:
#   -c CONFIG_PATH         Path to the configuration YAML file (default: config/config.yaml)
#   -k CHECKPOINT_PATH     Path to the model checkpoint file to resume training (optional)
#   -h                     Display this help message
#
# Example:
#   ./run_training.sh -c config/config.yaml -k checkpoints/checkpoint_epoch_10.pth
#
###############################################################################

set -euo pipefail

# Default configurations
CONFIG_PATH="config/config.yaml"
CHECKPOINT_PATH=""  # Empty by default; optional

# Function to display usage information
usage() {
    echo "Usage: $0 [-c CONFIG_PATH] [-k CHECKPOINT_PATH] [-h]"
    echo ""
    echo "Options:"
    echo "  -c CONFIG_PATH         Path to the configuration YAML file (default: config/config.yaml)"
    echo "  -k CHECKPOINT_PATH     Path to the model checkpoint file to resume training (optional)"
    echo "  -h                     Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 -c config/config.yaml -k checkpoints/checkpoint_epoch_10.pth"
    exit 1
}

# Parse command-line arguments using getopts
while getopts ":c:k:h" opt; do
    case "${opt}" in
        c)
            CONFIG_PATH="${OPTARG}"
            ;;
        k)
            CHECKPOINT_PATH="${OPTARG}"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            usage
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            usage
            ;;
    esac
done

shift $((OPTIND -1))

# Function to check if a file exists
check_file_exists() {
    local file_path="$1"
    if [[ ! -f "$file_path" ]]; then
        echo "Error: File '$file_path' does not exist."
        exit 1
    fi
}

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

# Optional: Download data
# Uncomment the following lines if data needs to be downloaded before training
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

# Create logs directory if it doesn't exist
mkdir -p logs

# Build the command to execute
TRAIN_COMMAND="python \"$TRAIN_SCRIPT\" --config \"$CONFIG_PATH\""
if [[ -n "$CHECKPOINT_PATH" ]]; then
    TRAIN_COMMAND+=" --checkpoint \"$CHECKPOINT_PATH\""
fi

# Execute the training script with the provided configuration and checkpoint
# Redirect both stdout and stderr to log file
eval $TRAIN_COMMAND > "logs/training_output.log" 2>&1

# Check the exit status of the training script
if [[ $? -ne 0 ]]; then
    echo "Model training failed. Check 'logs/training_output.log' for details."
    exit 1
else
    echo "Model training completed successfully."
fi

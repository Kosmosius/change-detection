#!/bin/bash

# scripts/run_evaluation.sh

###############################################################################
# Script to Run Model Evaluation
#
# This script orchestrates the evaluation process of a trained model. It ensures
# that the necessary data is available before performing evaluation. The evaluation
# results, including metrics, are logged for analysis.
#
# Usage:
#   ./run_evaluation.sh [-c CONFIG_PATH] [-k CHECKPOINT_PATH] [-h]
#
# Options:
#   -c CONFIG_PATH         Path to the configuration YAML file (default: config/config.yaml)
#   -k CHECKPOINT_PATH     Path to the model checkpoint file (default: checkpoints/best_model.pth)
#   -h                     Display this help message
#
# Example:
#   ./run_evaluation.sh -c config/config.yaml -k checkpoints/best_model.pth
#
###############################################################################

set -euo pipefail

# Default configurations
CONFIG_PATH="config/config.yaml"
CHECKPOINT_PATH="checkpoints/best_model.pth"

# Function to display usage information
usage() {
    echo "Usage: $0 [-c CONFIG_PATH] [-k CHECKPOINT_PATH] [-h]"
    echo ""
    echo "Options:"
    echo "  -c CONFIG_PATH         Path to the configuration YAML file (default: config/config.yaml)"
    echo "  -k CHECKPOINT_PATH     Path to the model checkpoint file (default: checkpoints/best_model.pth)"
    echo "  -h                     Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 -c config/config.yaml -k checkpoints/best_model.pth"
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

# Check if checkpoint file exists
check_file_exists "$CHECKPOINT_PATH"
echo "Checkpoint file '$CHECKPOINT_PATH' found."

# Optional: Activate virtual environment
# Uncomment and modify the following lines if using a virtual environment
# VENV_PATH="venv/bin/activate"
# if [[ -f "$VENV_PATH" ]]; then
#     source "$VENV_PATH"
#     echo "Activated virtual environment."
# else
#     echo "Warning: Virtual environment at '$VENV_PATH' not found. Proceeding without activation."
# fi

# Optional: Download and preprocess data
# Uncomment the following lines if data needs to be downloaded and preprocessed before evaluation
# echo "Starting data download..."
# python scripts/download_data.py --config "$CONFIG_PATH"
# echo "Data download completed."

# echo "Starting data preprocessing..."
# python scripts/preprocess_data.py --config "$CONFIG_PATH"
# echo "Data preprocessing completed."

# Run the evaluation
echo "Starting model evaluation..."

# Check if evaluate.py exists
EVALUATE_SCRIPT="evaluate/evaluate.py"
if [[ ! -f "$EVALUATE_SCRIPT" ]]; then
    echo "Error: Evaluation script '$EVALUATE_SCRIPT' not found."
    echo "Please ensure that 'evaluate/evaluate.py' exists and is executable."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Execute the evaluation script with the provided configuration and checkpoint
# Redirect both stdout and stderr to log file
python "$EVALUATE_SCRIPT" --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT_PATH" > "logs/evaluation_output.log" 2>&1

# Check the exit status of the evaluation script
if [[ $? -ne 0 ]]; then
    echo "Model evaluation failed. Check 'logs/evaluation_output.log' for details."
    exit 1
else
    echo "Model evaluation completed successfully."
fi

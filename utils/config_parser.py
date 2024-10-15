# utils/config_parser.py

import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)

def parse_config(config_path: str = "config/config.yaml") -> DictConfig:
    """
    Parses the configuration file and merges it with command-line arguments.

    Args:
        config_path (str): Path to the main configuration YAML file.

    Returns:
        DictConfig: Merged configuration object.
    """
    parser = argparse.ArgumentParser(description="Change Detection Project Configuration")
    
    # Define command-line arguments to override config settings
    parser.add_argument("--config", type=str, default=config_path, help="Path to the config file.")
    parser.add_argument("--model.name", type=str, help="Name of the model architecture.")
    parser.add_argument("--training.epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--training.learning_rate", type=float, help="Learning rate for the optimizer.")
    parser.add_argument("--data.train_data_path", type=str, help="Path to the training data.")
    parser.add_argument("--data.val_data_path", type=str, help="Path to the validation data.")
    # Add more arguments as needed
    
    args, unknown = parser.parse_known_args()
    
    # Load the main configuration file
    config = OmegaConf.load(args.config)
    
    # Convert argparse Namespace to a dictionary
    cli_args = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    
    # Merge command-line arguments into the configuration
    if cli_args:
        config = OmegaConf.merge(config, cli_args)
        logger.info("Command-line arguments have been merged into the configuration.")
    else:
        logger.info("No command-line arguments provided to override the configuration.")
    
    # Optionally, validate the configuration here
    # For example, ensure required fields are present
    required_fields = ["model.name", "training.epochs", "training.learning_rate", "data.train_data_path", "data.val_data_path"]
    missing_fields = [field for field in required_fields if OmegaConf.select(config, field) is None]
    if missing_fields:
        logger.error(f"Missing required configuration fields: {missing_fields}")
        raise ValueError(f"Missing required configuration fields: {missing_fields}")
    
    return config

def save_config(config: DictConfig, save_path: str):
    """
    Saves the current configuration to a YAML file.

    Args:
        config (DictConfig): Configuration object to save.
        save_path (str): Path where the configuration file will be saved.
    """
    save_path = Path(save_path)
    try:
        OmegaConf.save(config, save_path)
        logger.info(f"Configuration saved to '{save_path}'.")
    except Exception as e:
        logger.error(f"Failed to save configuration to '{save_path}': {e}")

def get_config():
    """
    Entry point for configuration parsing.

    Returns:
        DictConfig: Parsed and merged configuration.
    """
    config = parse_config()
    return config

if __name__ == "__main__":
    config = get_config()
    print(OmegaConf.to_yaml(config))

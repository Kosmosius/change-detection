# utils/config_parser.py

import argparse
import sys
from pathlib import Path
from typing import List

from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


def parse_config(config_path: str = "config/config.yaml") -> DictConfig:
    """
    Parses the configuration file and merges it with command-line overrides.
    Allows overriding any config parameter using dot notation in the command line.
    
    Example Overrides:
        --model.name Transformer
        --training.epochs 50
        --data.use_s3 True

    Args:
        config_path (str): Path to the main configuration YAML file.

    Returns:
        DictConfig: Merged configuration object.

    Raises:
        ValueError: If required configuration fields are missing after merging.
    """
    parser = argparse.ArgumentParser(
        description="Change Detection Project Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argument to specify config file
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the configuration YAML file."
    )
    
    # Capture all other arguments as overrides using a custom syntax
    # e.g., --model.name Transformer --training.epochs 50
    parser.add_argument(
        'overrides',
        nargs=argparse.REMAINDER,
        help="Override configuration parameters using key=value syntax."
    )
    
    args = parser.parse_args()
    
    # Load the main configuration file
    try:
        config = OmegaConf.load(args.config)
        logger.info(f"Configuration loaded from '{args.config}'.")
    except Exception as e:
        logger.error(f"Failed to load configuration file '{args.config}': {e}")
        sys.exit(1)
    
    # Parse overrides
    if args.overrides:
        # Convert overrides list to a single string
        overrides_str = ' '.join(args.overrides)
        try:
            # OmegaConf can parse command-line overrides using from_cli
            override_conf = OmegaConf.from_cli(overrides_str.split())
            config = OmegaConf.merge(config, override_conf)
            logger.info("Command-line overrides have been merged into the configuration.")
        except Exception as e:
            logger.error(f"Failed to parse command-line overrides: {e}")
            sys.exit(1)
    else:
        logger.info("No command-line overrides provided.")
    
    # Validate configuration
    validate_config(config)
    
    return config


def validate_config(config: DictConfig):
    """
    Validates the configuration to ensure all required fields are present.

    Args:
        config (DictConfig): Configuration object to validate.

    Raises:
        ValueError: If any required configuration fields are missing.
    """
    required_fields = [
        "model.name",
        "training.epochs",
        "training.learning_rate",
        "data.train_image_pairs",
        "data.train_labels",
        "data.val_image_pairs",
        "data.val_labels",
        "data.use_s3",
        "data.s3_bucket",
        "data.s3_prefix",
        "training.batch_size",
        "training.loss_function",
        "training.metrics",
        "metrics.threshold",
        # Add more required fields as necessary
    ]
    
    missing_fields = [field for field in required_fields if OmegaConf.select(config, field) is None]
    if missing_fields:
        logger.error(f"Missing required configuration fields: {missing_fields}")
        raise ValueError(f"Missing required configuration fields: {missing_fields}")
    else:
        logger.info("All required configuration fields are present.")


def save_config(config: DictConfig, save_path: str):
    """
    Saves the current configuration to a YAML file.

    Args:
        config (DictConfig): Configuration object to save.
        save_path (str): Path where the configuration file will be saved.
    
    Raises:
        OSError: If the configuration file cannot be saved.
    """
    save_path = Path(save_path)
    try:
        OmegaConf.save(config, save_path)
        logger.info(f"Configuration saved to '{save_path}'.")
    except Exception as e:
        logger.error(f"Failed to save configuration to '{save_path}': {e}")
        raise OSError(f"Failed to save configuration to '{save_path}': {e}")


def get_config() -> DictConfig:
    """
    Entry point for configuration parsing.

    Returns:
        DictConfig: Parsed and merged configuration.
    """
    config = parse_config()
    return config


if __name__ == "__main__":
    """
    If executed as a script, parse and print the merged configuration.
    """
    try:
        config = get_config()
        print(OmegaConf.to_yaml(config))
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        sys.exit(1)

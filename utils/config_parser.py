# utils/config_parser.py

import argparse
import sys
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import OmegaConfBaseException
import logging

logger = logging.getLogger(__name__)


def parse_config(config_path: str = "config/config.yaml") -> DictConfig:
    """
    Parses the configuration file and merges it with command-line overrides.
    Allows overriding any config parameter using dot notation in the command line.

    Example Overrides:
        --model.name=Transformer
        --training.epochs=50
        --data.use_s3=True

    Args:
        config_path (str): Path to the main configuration YAML file.

    Returns:
        DictConfig: Merged configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        OmegaConfBaseException: If the configuration file cannot be parsed.
        ValueError: If required configuration fields are missing after merging.
    """
    parser = argparse.ArgumentParser(
        description="Change Detection Project Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False
    )

    # Argument to specify config file
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the configuration YAML file."
    )

    # Parse known args to separate config file argument and overrides
    args, unknown_args = parser.parse_known_args()

    # Load the main configuration file
    try:
        config = OmegaConf.load(args.config)
        logger.info("Configuration loaded from '%s'.", args.config)
    except FileNotFoundError as e:
        logger.error("Configuration file '%s' not found: %s", args.config, e)
        sys.exit(1)
    except OmegaConfBaseException as e:
        logger.error("Error parsing configuration file '%s': %s", args.config, e)
        sys.exit(1)
    except (OSError, IOError) as e:
        logger.error("Failed to load configuration file '%s': %s", args.config, e)
        sys.exit(1)

    # Parse overrides
    if unknown_args:
        try:
            # OmegaConf can parse command-line overrides using from_cli
            override_conf = OmegaConf.from_cli(unknown_args)
            config = OmegaConf.merge(config, override_conf)
            logger.info("Command-line overrides have been merged into the configuration.")
        except OmegaConfBaseException as e:
            logger.error("Failed to parse command-line overrides: %s", e)
            sys.exit(1)
    else:
        logger.info("No command-line overrides provided.")

    # Validate configuration
    validate_config(config)

    return config


def validate_config(config: DictConfig) -> None:
    """
    Validates the configuration to ensure all required fields are present.

    Args:
        config (DictConfig): Configuration object to validate.

    Raises:
        ValueError: If any required configuration fields are missing.
    """
    required_fields = [
        "model.name",
        "training.num_epochs",
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
        missing_fields_str = ', '.join(missing_fields)
        logger.error("Missing required configuration fields: %s", missing_fields_str)
        raise ValueError(f"Missing required configuration fields: {missing_fields_str}")
    else:
        logger.info("All required configuration fields are present.")


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Saves the current configuration to a YAML file.

    Args:
        config (DictConfig): Configuration object to save.
        save_path (Union[str, Path]): Path where the configuration file will be saved.

    Raises:
        OSError: If the configuration file cannot be saved.
        IOError: If an I/O operation fails.
    """
    save_path = Path(save_path)
    try:
        OmegaConf.save(config, save_path)
        logger.info("Configuration saved to '%s'.", save_path)
    except (OSError, IOError) as e:
        logger.error("Failed to save configuration to '%s': %s", save_path, e)
        raise


def get_config() -> DictConfig:
    """
    Entry point for configuration parsing.

    Returns:
        DictConfig: Parsed and merged configuration.

    Raises:
        ValueError: If configuration parsing fails.
    """
    try:
        config = parse_config()
        return config
    except ValueError as e:
        logger.error("Configuration parsing failed: %s", e)
        raise


if __name__ == "__main__":
    """
    If executed as a script, parse and print the merged configuration.
    """
    try:
        config = get_config()
        print(OmegaConf.to_yaml(config))
    except (ValueError, OSError, IOError) as e:
        logger.error("Failed to get configuration: %s", e)
        sys.exit(1)

# utils/huggingface_utils.py

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, LoraConfig

logger = logging.getLogger(__name__)


def load_huggingface_model(
    model_name: str,
    pretrained: bool = True,
    config_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> PreTrainedModel:
    """
    Loads a HuggingFace model.

    Args:
        model_name (str): The name or path of the HuggingFace model.
        pretrained (bool): Whether to load pre-trained weights. Defaults to True.
        config_kwargs (Optional[Dict[str, Any]]): Additional configuration parameters.
        model_kwargs (Optional[Dict[str, Any]]): Additional model parameters.

    Returns:
        PreTrainedModel: The loaded HuggingFace model.

    Raises:
        FileNotFoundError: If the model files are not found.
        ValueError: If invalid arguments are provided.
        RuntimeError: If loading the model fails.
    """
    config_kwargs = config_kwargs or {}
    model_kwargs = model_kwargs or {}

    try:
        if pretrained:
            config = AutoConfig.from_pretrained(model_name, **config_kwargs)
            model = AutoModel.from_pretrained(model_name, config=config, **model_kwargs)
        else:
            config = AutoConfig(**config_kwargs)
            model = AutoModel(config, **model_kwargs)
        logger.info("HuggingFace model '%s' loaded successfully.", model_name)
        return model
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Error loading HuggingFace model '%s': %s", model_name, e)
        raise RuntimeError(f"Failed to load HuggingFace model '{model_name}': {e}") from e


def apply_peft(
    model: PreTrainedModel,
    peft_config: Dict[str, Any],
    peft_model_name: Optional[str] = None,
) -> PeftModel:
    """
    Applies Parameter-Efficient Fine-Tuning (PEFT) to a HuggingFace model.

    Args:
        model (PreTrainedModel): The HuggingFace model to fine-tune.
        peft_config (Dict[str, Any]): Configuration dictionary for PEFT.
        peft_model_name (Optional[str]): Name or path of the PEFT model to load.

    Returns:
        PeftModel: The PEFT-enhanced model.

    Raises:
        KeyError: If required keys are missing in peft_config.
        ValueError: If invalid configuration is provided.
        RuntimeError: If applying PEFT fails.
    """
    required_keys = {"task_type", "r", "lora_alpha", "lora_dropout", "target_modules"}
    missing_keys = required_keys - peft_config.keys()
    if missing_keys:
        logger.error("PEFT configuration missing keys: %s", ', '.join(missing_keys))
        raise KeyError(f"PEFT configuration missing keys: {', '.join(missing_keys)}")

    try:
        lora_config = LoraConfig(
            task_type=peft_config["task_type"],
            r=peft_config["r"],
            lora_alpha=peft_config["lora_alpha"],
            lora_dropout=peft_config["lora_dropout"],
            target_modules=peft_config["target_modules"],
        )
        if peft_model_name:
            peft_model = PeftModel.from_pretrained(model, peft_model_name, config=lora_config)
            logger.info("PEFT model loaded from '%s' and applied successfully.", peft_model_name)
        else:
            peft_model = PeftModel(model, lora_config)
            logger.info("PEFT applied to the model successfully.")
        return peft_model
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        logger.error("Error applying PEFT: %s", e)
        raise RuntimeError(f"Failed to apply PEFT: {e}") from e


def integrate_huggingface_with_custom_model(
    custom_model: torch.nn.Module,
    hf_model: PreTrainedModel,
    encoder_attr: str = "encoder",
) -> torch.nn.Module:
    """
    Integrates a HuggingFace model into a custom model architecture.

    Args:
        custom_model (torch.nn.Module): The custom model to integrate with.
        hf_model (PreTrainedModel): The HuggingFace model to integrate.
        encoder_attr (str): Attribute name in the custom model where the HuggingFace model should be integrated.

    Returns:
        torch.nn.Module: The integrated model.

    Raises:
        AttributeError: If the custom model does not have the specified encoder attribute or if setting the attribute fails.
    """
    if not hasattr(custom_model, encoder_attr):
        logger.error("Custom model does not have an attribute '%s'.", encoder_attr)
        raise AttributeError(f"Custom model does not have an attribute '{encoder_attr}'.")

    try:
        setattr(custom_model, encoder_attr, hf_model)
        logger.info("HuggingFace model integrated into '%s' attribute of the custom model successfully.", encoder_attr)
        return custom_model
    except AttributeError as e:
        logger.error("Error integrating HuggingFace model into custom model: %s", e)
        raise


def save_huggingface_model(
    model: PreTrainedModel,
    save_directory: Union[str, Path],
    model_name: str = "custom_model",
    save_config: bool = True,
) -> None:
    """
    Saves the HuggingFace model to the specified directory.

    Args:
        model (PreTrainedModel): The HuggingFace model to save.
        save_directory (Union[str, Path]): The directory where the model will be saved.
        model_name (str): The name of the saved model directory.
        save_config (bool): Whether to save the model configuration.

    Raises:
        OSError: If saving the model fails due to an OS error.
        IOError: If an I/O operation fails.
    """
    save_directory = Path(save_directory)
    save_path = save_directory / model_name
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path, save_config=save_config)
        logger.info("HuggingFace model saved to '%s'.", save_path)
    except (OSError, IOError) as e:
        logger.error("Error saving HuggingFace model to '%s': %s", save_path, e)
        raise


def load_peft_model(
    model_path: Union[str, Path],
    peft_config: Optional[Dict[str, Any]] = None,
) -> PeftModel:
    """
    Loads a PEFT-enhanced HuggingFace model from the specified path.

    Args:
        model_path (Union[str, Path]): The path to the PEFT model directory.
        peft_config (Optional[Dict[str, Any]]): Configuration dictionary for PEFT.

    Returns:
        PeftModel: The loaded PEFT model.

    Raises:
        FileNotFoundError: If the model directory does not exist.
        KeyError: If required keys are missing in peft_config.
        ValueError: If invalid configuration is provided.
        RuntimeError: If loading the PEFT model fails.
    """
    model_path = Path(model_path)
    if not model_path.is_dir():
        logger.error("PEFT model directory '%s' does not exist.", model_path)
        raise FileNotFoundError(f"PEFT model directory '{model_path}' does not exist.")

    try:
        if peft_config:
            required_keys = {"task_type", "r", "lora_alpha", "lora_dropout", "target_modules"}
            missing_keys = required_keys - peft_config.keys()
            if missing_keys:
                logger.error("PEFT configuration missing keys: %s", ', '.join(missing_keys))
                raise KeyError(f"PEFT configuration missing keys: {', '.join(missing_keys)}")

            lora_config = LoraConfig(
                task_type=peft_config["task_type"],
                r=peft_config["r"],
                lora_alpha=peft_config["lora_alpha"],
                lora_dropout=peft_config["lora_dropout"],
                target_modules=peft_config["target_modules"],
            )
            peft_model = PeftModel.from_pretrained(model_path, config=lora_config)
            logger.info("PEFT model loaded from '%s' with provided configuration.", model_path)
        else:
            peft_model = PeftModel.from_pretrained(model_path)
            logger.info("PEFT model loaded from '%s' successfully.", model_path)
        return peft_model
    except (OSError, IOError, ValueError, RuntimeError, KeyError, TypeError) as e:
        logger.error("Error loading PEFT model from '%s': %s", model_path, e)
        raise


def get_huggingface_tokenizer(
    tokenizer_name: str,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
) -> PreTrainedTokenizer:
    """
    Loads a HuggingFace tokenizer.

    Args:
        tokenizer_name (str): The name or path of the HuggingFace tokenizer.
        tokenizer_kwargs (Optional[Dict[str, Any]]): Additional arguments to pass to the tokenizer constructor.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.

    Raises:
        FileNotFoundError: If the tokenizer files are not found.
        ValueError: If invalid arguments are provided.
        RuntimeError: If loading the tokenizer fails.
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
        logger.info("HuggingFace tokenizer '%s' loaded successfully.", tokenizer_name)
        return tokenizer
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Error loading HuggingFace tokenizer '%s': %s", tokenizer_name, e)
        raise

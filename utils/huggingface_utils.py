# utils/huggingface_utils.py

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, LoraConfig, TaskType
import torch

logger = logging.getLogger(__name__)


def load_huggingface_model(
    model_name: str,
    pretrained: bool = True,
    config_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> PreTrainedModel:
    """
    Loads a pre-trained HuggingFace model.

    Args:
        model_name (str): The name or path of the HuggingFace model.
        pretrained (bool): Whether to load pre-trained weights. Defaults to True.
        config_kwargs (Dict[str, Any], optional): Additional configuration parameters.
        model_kwargs (Dict[str, Any], optional): Additional model parameters.

    Returns:
        PreTrainedModel: The loaded HuggingFace model.

    Raises:
        ValueError: If the model fails to load.
    """
    config_kwargs = config_kwargs or {}
    model_kwargs = model_kwargs or {}
    try:
        config = AutoConfig.from_pretrained(model_name, **config_kwargs) if pretrained else AutoConfig(**config_kwargs)
        model = AutoModel.from_pretrained(model_name, config=config, **model_kwargs) if pretrained else AutoModel(config, **model_kwargs)
        logger.info(f"HuggingFace model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading HuggingFace model '{model_name}': {e}")
        raise ValueError(f"Failed to load HuggingFace model '{model_name}'.") from e


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
        peft_model_name (str, optional): Name or path of the PEFT model to load. Required if applying from a pretrained PEFT model.

    Returns:
        PeftModel: The PEFT-enhanced model.

    Raises:
        ValueError: If PEFT application fails due to missing parameters or incorrect configurations.
    """
    required_keys = {"task_type", "r", "lora_alpha", "lora_dropout", "target_modules"}
    if not required_keys.issubset(peft_config.keys()):
        missing = required_keys - peft_config.keys()
        logger.error(f"PEFT configuration missing keys: {missing}")
        raise ValueError(f"PEFT configuration missing keys: {missing}")

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
            logger.info(f"PEFT model loaded from '{peft_model_name}' and applied successfully.")
        else:
            peft_model = PeftModel(model, lora_config)
            logger.info("PEFT applied to the model successfully.")
        return peft_model
    except Exception as e:
        logger.error(f"Error applying PEFT: {e}")
        raise ValueError("Failed to apply PEFT to the model.") from e


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
        encoder_attr (str, optional): Attribute name in the custom model where the HuggingFace model should be integrated. Defaults to "encoder".

    Returns:
        torch.nn.Module: The integrated model.

    Raises:
        AttributeError: If the custom model does not have the specified encoder attribute.
    """
    if not hasattr(custom_model, encoder_attr):
        logger.error(f"Custom model does not have an attribute '{encoder_attr}'.")
        raise AttributeError(f"Custom model does not have an attribute '{encoder_attr}'.")
    
    try:
        setattr(custom_model, encoder_attr, hf_model)
        logger.info(f"HuggingFace model integrated into '{encoder_attr}' attribute of the custom model successfully.")
        return custom_model
    except Exception as e:
        logger.error(f"Error integrating HuggingFace model into custom model: {e}")
        raise ValueError("Failed to integrate HuggingFace model into custom model.") from e


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
        save_directory (str or Path): The directory where the model will be saved.
        model_name (str, optional): The name of the saved model directory. Defaults to "custom_model".
        save_config (bool, optional): Whether to save the model configuration. Defaults to True.

    Raises:
        ValueError: If saving the model fails.
    """
    save_directory = Path(save_directory)
    save_path = save_directory / model_name
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path, save_config=save_config)
        logger.info(f"HuggingFace model saved to '{save_path}'.")
    except Exception as e:
        logger.error(f"Error saving HuggingFace model to '{save_path}': {e}")
        raise ValueError(f"Failed to save HuggingFace model to '{save_path}'.") from e


def load_peft_model(
    model_path: Union[str, Path],
    peft_config: Optional[Dict[str, Any]] = None,
) -> PeftModel:
    """
    Loads a PEFT-enhanced HuggingFace model from the specified path.

    Args:
        model_path (str or Path): The path to the PEFT model directory.
        peft_config (Dict[str, Any], optional): Configuration dictionary for PEFT. Required if not loading from a pretrained PEFT model.

    Returns:
        PeftModel: The loaded PEFT model.

    Raises:
        ValueError: If loading the PEFT model fails.
    """
    model_path = Path(model_path)
    if not model_path.is_dir():
        logger.error(f"PEFT model directory '{model_path}' does not exist.")
        raise ValueError(f"PEFT model directory '{model_path}' does not exist.")
    
    try:
        if peft_config:
            lora_config = LoraConfig(
                task_type=peft_config["task_type"],
                r=peft_config["r"],
                lora_alpha=peft_config["lora_alpha"],
                lora_dropout=peft_config["lora_dropout"],
                target_modules=peft_config["target_modules"],
            )
            peft_model = PeftModel.from_pretrained(model_path, config=lora_config)
            logger.info(f"PEFT model loaded from '{model_path}' with provided configuration.")
        else:
            peft_model = PeftModel.from_pretrained(model_path)
            logger.info(f"PEFT model loaded from '{model_path}' successfully.")
        return peft_model
    except Exception as e:
        logger.error(f"Error loading PEFT model from '{model_path}': {e}")
        raise ValueError(f"Failed to load PEFT model from '{model_path}'.") from e


def get_huggingface_tokenizer(
    tokenizer_name: str,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
) -> PreTrainedTokenizer:
    """
    Loads a HuggingFace tokenizer.

    Args:
        tokenizer_name (str): The name or path of the HuggingFace tokenizer.
        tokenizer_kwargs (Dict[str, Any], optional): Additional arguments to pass to the tokenizer constructor.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.

    Raises:
        ValueError: If the tokenizer fails to load.
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
        logger.info(f"HuggingFace tokenizer '{tokenizer_name}' loaded successfully.")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading HuggingFace tokenizer '{tokenizer_name}': {e}")
        raise ValueError(f"Failed to load HuggingFace tokenizer '{tokenizer_name}'.") from e

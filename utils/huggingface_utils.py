# utils/huggingface_utils.py

import logging
from pathlib import Path
from transformers import AutoModel, AutoConfig
from peft import PeftModel, LoraConfig, TaskType
import torch

logger = logging.getLogger(__name__)

def load_huggingface_model(model_name: str, pretrained: bool = True, **kwargs) -> torch.nn.Module:
    """
    Loads a pre-trained HuggingFace model.

    Args:
        model_name (str): The name or path of the HuggingFace model.
        pretrained (bool): Whether to load pre-trained weights.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        torch.nn.Module: The loaded HuggingFace model.
    """
    try:
        config = AutoConfig.from_pretrained(model_name) if pretrained else AutoConfig()
        model = AutoModel.from_pretrained(model_name, config=config) if pretrained else AutoModel(config)
        logger.info(f"HuggingFace model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading HuggingFace model '{model_name}': {e}")
        raise

def apply_peft(model: torch.nn.Module, peft_config: dict) -> PeftModel:
    """
    Applies Parameter-Efficient Fine-Tuning (PEFT) to a HuggingFace model.

    Args:
        model (torch.nn.Module): The HuggingFace model to fine-tune.
        peft_config (dict): Configuration dictionary for PEFT.

    Returns:
        PeftModel: The PEFT-enhanced model.
    """
    try:
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Adjust based on the task, e.g., IMAGE_CLASSIFICATION
            r=peft_config.get('r', 8),
            lora_alpha=peft_config.get('lora_alpha', 32),
            lora_dropout=peft_config.get('lora_dropout', 0.1),
            target_modules=peft_config.get('target_modules', ["query", "key", "value"])
        )
        peft_model = PeftModel.from_pretrained(model, peft_config.get('peft_model_name'), config=config)
        logger.info("PEFT applied to the model successfully.")
        return peft_model
    except Exception as e:
        logger.error(f"Error applying PEFT: {e}")
        raise

def integrate_huggingface_with_custom_model(custom_model: torch.nn.Module, hf_model: torch.nn.Module) -> torch.nn.Module:
    """
    Integrates a HuggingFace model into a custom model architecture.

    Args:
        custom_model (torch.nn.Module): The custom model to integrate with.
        hf_model (torch.nn.Module): The HuggingFace model to integrate.

    Returns:
        torch.nn.Module: The integrated model.
    """
    try:
        # Example: Replace the encoder of the custom model with the HuggingFace model
        if hasattr(custom_model, 'encoder'):
            custom_model.encoder = hf_model
            logger.info("HuggingFace model integrated into the custom model successfully.")
        else:
            logger.warning("Custom model does not have an 'encoder' attribute. Integration skipped.")
        return custom_model
    except Exception as e:
        logger.error(f"Error integrating HuggingFace model with custom model: {e}")
        raise

def save_huggingface_model(model: torch.nn.Module, save_directory: str, model_name: str = "custom_model"):
    """
    Saves the HuggingFace model to the specified directory.

    Args:
        model (torch.nn.Module): The HuggingFace model to save.
        save_directory (str): The directory where the model will be saved.
        model_name (str): The name of the saved model directory.
    """
    save_path = Path(save_directory) / model_name
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        logger.info(f"HuggingFace model saved to '{save_path}'.")
    except Exception as e:
        logger.error(f"Error saving HuggingFace model to '{save_path}': {e}")
        raise

def load_peft_model(model_path: str) -> PeftModel:
    """
    Loads a PEFT-enhanced HuggingFace model from the specified path.

    Args:
        model_path (str): The path to the PEFT model directory.

    Returns:
        PeftModel: The loaded PEFT model.
    """
    try:
        peft_model = PeftModel.from_pretrained(model_path)
        logger.info(f"PEFT model loaded from '{model_path}'.")
        return peft_model
    except Exception as e:
        logger.error(f"Error loading PEFT model from '{model_path}': {e}")
        raise

def get_huggingface_tokenizer(tokenizer_name: str, **kwargs):
    """
    Loads a HuggingFace tokenizer.

    Args:
        tokenizer_name (str): The name or path of the HuggingFace tokenizer.
        **kwargs: Additional arguments to pass to the tokenizer constructor.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.
    """
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
        logger.info(f"HuggingFace tokenizer '{tokenizer_name}' loaded successfully.")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading HuggingFace tokenizer '{tokenizer_name}': {e}")
        raise

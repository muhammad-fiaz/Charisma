"""Model manager for loading and configuring models"""

from typing import Tuple, Optional

try:
    import torch
    from unsloth import FastLanguageModel
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.config.models import AVAILABLE_MODELS
from charisma.utils.logger import get_logger

logger = get_logger()


class ModelManager:
    """Manages model loading and configuration with Unsloth"""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        use_lora: bool = True,
        lora_config: Optional[dict] = None,
    ) -> Tuple:
        """Load model and tokenizer using Unsloth FastModel"""
        logger.info("Loading model: %s", model_name)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detection
            load_in_4bit=load_in_4bit,
        )

        if use_lora:
            self._apply_lora(lora_config or {})

        logger.info("Model loaded successfully")
        return self.model, self.tokenizer

    def _apply_lora(self, lora_config: dict) -> None:
        """Apply LoRA adapters to model"""
        target_modules = lora_config.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        r = lora_config.get("r", 16)
        lora_alpha = lora_config.get("lora_alpha", 16)
        lora_dropout = lora_config.get("lora_dropout", 0)
        use_gradient_checkpointing = lora_config.get(
            "use_gradient_checkpointing", "unsloth"
        )
        use_rslora = lora_config.get("use_rslora", False)
        use_loftq = lora_config.get("use_loftq", False)

        # Prepare loftq_config if enabled
        loftq_config = None
        if use_loftq:
            loftq_config = {"loftq_bits": 4}

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=3407,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

        logger.info("LoRA adapters applied - r=%d, alpha=%d, rslora=%s, loftq=%s", 
                    r, lora_alpha, use_rslora, use_loftq)

    def prepare_for_inference(self):
        """Prepare model for fast inference"""
        FastLanguageModel.for_inference(self.model)
        logger.info("Model prepared for inference")

    def get_model_info(self, model_name: str) -> dict:
        """Get information about a model"""
        for model in AVAILABLE_MODELS:
            if model["id"] == model_name:
                return model

        return {
            "id": model_name,
            "name": model_name.split("/")[-1],
            "description": "Custom model",
            "vram": "Unknown",
            "params": "Unknown",
        }

    def save_model(self, output_dir: str, save_method: str = "lora") -> None:
        """Save model to disk"""
        logger.info("Saving model to: %s (method=%s)", output_dir, save_method)

        if save_method == "lora":
            self.model.save_pretrained(output_dir, max_shard_size="5GB")
            self.tokenizer.save_pretrained(output_dir)
        elif save_method == "merged_16bit":
            self.model.save_pretrained_merged(
                output_dir, self.tokenizer, save_method="merged_16bit", max_shard_size="5GB"
            )
        elif save_method == "merged_4bit":
            self.model.save_pretrained_merged(
                output_dir, self.tokenizer, save_method="merged_4bit", max_shard_size="5GB"
            )
        else:
            raise ValueError(f"Invalid save method: {save_method}")

        logger.info("Model saved successfully")

    def get_vram_usage(self) -> dict:
        """Get current VRAM usage"""
        if not torch.cuda.is_available():
            return {"available": False}

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return {
            "available": True,
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - reserved, 2),
        }

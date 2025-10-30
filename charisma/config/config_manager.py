"""Configuration manager for Charisma - Handles reading and writing to charisma.toml"""

import toml
from pathlib import Path
from typing import Any, Dict
from charisma.utils.logger import get_logger
from charisma.__version__ import __version__

logger = get_logger()


class ConfigManager:
    """Manages configuration file operations"""

    def __init__(self, config_path: str = "charisma.toml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_path.exists():
            logger.warning(
                f"Config file not found: {self.config_path}. Creating default config."
            )
            self._create_default_config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = toml.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._create_default_config()

        return self.config

    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                toml.dump(self.config, f)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Update entire configuration section"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(values)

    def _create_default_config(self) -> None:
        """Create default configuration file"""
        default_config = {
            "project": {
                "name": "charisma",
                "version": __version__,
                "repository": "muhammad-fiaz/charisma",
                "email": "contact@muhammadfiaz.com",
            },
            "model": {
                "default_model": "unsloth/gemma-3-270m-it",
                "max_seq_length": 2048,
                "load_in_4bit": False,
                "load_in_8bit": False,
            },
            "training": {
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 5,
                "max_steps": 100,
                "learning_rate": 0.00005,
                "weight_decay": 0.01,
                "lr_scheduler_type": "linear",
                "optim": "adamw_8bit",
                "logging_steps": 1,
                "output_dir": "outputs",
                "seed": 3407,
            },
            "lora": {
                "r": 128,
                "lora_alpha": 128,
                "lora_dropout": 0,
                "bias": "none",
                "use_gradient_checkpointing": "unsloth",
                "use_rslora": False,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "inference": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "max_new_tokens": 125,
            },
            "notion": {
                "api_key": "",
                "oauth_client_id": "",
                "oauth_client_secret": "",
            },
            "huggingface": {
                "api_token": "",
                "default_repo": "",
            },
            "ui": {
                "server_name": "127.0.0.1",
                "server_port": 7860,
                "share": False,
            },
        }

        self.config = default_config
        self.save_config()


# Global configuration instance
_config_instance = None


def get_config(config_path: str = "charisma.toml"):
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance

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
        """Load configuration from file, merging with defaults"""
        # Get default config structure
        default_config = self._get_default_config()
        
        if not self.config_path.exists():
            logger.warning(
                f"Config file not found: {self.config_path}. Creating default config."
            )
            self.config = default_config
            self.save_config()
        else:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded_config = toml.load(f)
                
                # Merge loaded config with defaults (preserves user values, adds missing defaults)
                self.config = self._merge_configs(default_config, loaded_config)
                
                # Save merged config to update file with any new fields
                self.save_config()
                
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using default config.")
                self.config = default_config
                self.save_config()

        return self.config

    def _merge_configs(self, defaults: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with defaults, preserving user values"""
        result = defaults.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Use loaded value (user's config takes precedence)
                result[key] = value
        
        return result

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

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure with all settings"""
        return {
            "project": {
                "name": "charisma",
                "version": __version__,
                "repository": "muhammad-fiaz/charisma",
                "email": "contact@muhammadfiaz.com",
            },
            "model": {
                "default_model": "unsloth/gemma-3-270m-it",
                "max_seq_length": 2048,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "use_cache": True,
            },
            "training": {
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": 0.0002,
                "num_epochs": 1,
                "max_steps": 60,
                "warmup_steps": 5,
                "logging_steps": 1,
                "optimizer": "adamw_8bit",
                "weight_decay": 0.01,
                "lr_scheduler_type": "linear",
            },
            "lora": {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "bias": "none",
                "use_gradient_checkpointing": "unsloth",
                "use_rslora": False,
                "use_loftq": False,
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
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_p": 0.9,
                "use_cache": True,
            },
            "notion": {
                "api_key": "",
                "oauth_client_id": "",
                "oauth_client_secret": "",
            },
            "huggingface": {
                "token": "",
                "default_repo": "my-charisma-model",
                "private": True,
            },
            "ui": {
                "server_name": "127.0.0.1",
                "server_port": 7860,
                "share": False,
            },
            "system": {
                "debug_logs": False,
                "num_gpus": 1,
                "gpu_ids": "0",
            },
            "unsloth": {
                "dataset_num_proc": 1,
                "packing": False,
                "use_gradient_checkpointing": True,
                "use_rslora": False,
                "use_loftq": False,
            },
            "prompts": {
                "system_prompt": """You are an AI clone trained to act, think, and respond exactly like {name}.

Your personality and characteristics:
- Name: {name}
- Age: {age}
- Location: {location}, {country}
- Interests: {hobbies}
- Favorites: {favorites}

About you:
{bio}

Based on the memories and experiences you've been trained on, you should:
1. Respond in the same tone and style as {name}
2. Use similar vocabulary and expressions
3. Reference memories and past experiences naturally
4. Maintain consistent personality traits
5. Show the same interests and preferences

Always stay in character and respond as {name} would respond.""",
                "response_structure": """When responding:
1. Be natural and conversational
2. Draw from relevant memories when appropriate
3. Stay true to the personality and characteristics
4. Keep responses authentic and personal
5. Use first-person perspective (I, me, my)""",
            },
        }


# Global configuration instance
_config_instance = None


def get_config(config_path: str = "charisma.toml"):
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance

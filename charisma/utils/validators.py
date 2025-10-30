"""Input validation utilities for Charisma"""

import re
from typing import Dict, List, Tuple


def validate_notion_token(token: str) -> Tuple[bool, str]:
    """Validate Notion API token format"""
    if not token:
        return False, "Notion token cannot be empty"
    if not token.startswith("secret_") and not token.startswith("ntn_"):
        return False, "Notion token must start with 'secret_' or 'ntn_'"
    if len(token) < 20:
        return False, "Notion token appears to be too short"
    return True, ""


def validate_hf_token(token: str) -> Tuple[bool, str]:
    """Validate HuggingFace API token format"""
    if not token:
        return False, "HuggingFace token cannot be empty"
    if not token.startswith("hf_"):
        return False, "HuggingFace token must start with 'hf_'"
    if len(token) < 20:
        return False, "HuggingFace token appears to be too short"
    return True, ""


def validate_model_name(model_name: str) -> Tuple[bool, str]:
    """Validate model name format"""
    if not model_name:
        return False, "Model name cannot be empty"
    if "/" not in model_name:
        return False, "Model name must be in format 'username/model-name'"
    parts = model_name.split("/")
    if len(parts) != 2:
        return False, "Model name must have exactly one '/' separator"
    username, repo = parts
    if not username or not repo:
        return False, "Username and repository name cannot be empty"
    pattern = r"^[a-zA-Z0-9][-a-zA-Z0-9_.]*$"
    if not re.match(pattern, username) or not re.match(pattern, repo):
        return False, "Model name contains invalid characters"
    return True, ""


def validate_personal_info(info: Dict[str, str]) -> Tuple[bool, List[str]]:
    """Validate personal information fields"""
    errors = []
    required_fields = ["name"]

    for field in required_fields:
        if field not in info or not info[field].strip():
            errors.append(f"'{field}' is required")

    if "age" in info and info["age"]:
        try:
            age = int(info["age"])
            if age < 0 or age > 150:
                errors.append("Age must be between 0 and 150")
        except ValueError:
            errors.append("Age must be a valid number")

    if "name" in info and info["name"]:
        if len(info["name"]) < 2:
            errors.append("Name must be at least 2 characters long")
        if len(info["name"]) > 100:
            errors.append("Name must be less than 100 characters")

    return len(errors) == 0, errors


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    filename = re.sub(r'[<>:"/\\|?*]', "", filename)
    filename = filename.replace(" ", "_")
    filename = re.sub(r"_{2,}", "_", filename)
    if len(filename) > 200:
        filename = filename[:200]
    return filename

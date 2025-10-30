"""Available models configuration for Charisma"""

# Available pre-configured models
AVAILABLE_MODELS = [
    {
        "id": "unsloth/gemma-3-270m-it",
        "name": "Gemma-3 270M",
        "description": "Fast & efficient, great for beginners",
        "vram": "~4GB",
        "params": "270M",
    },
    {
        "id": "unsloth/gemma-2-2b-it",
        "name": "Gemma-2 2B",
        "description": "Balanced performance and quality",
        "vram": "~6GB",
        "params": "2B",
    },
    {
        "id": "unsloth/Llama-3.2-1B-Instruct",
        "name": "Llama-3.2 1B",
        "description": "Compact and fast",
        "vram": "~4GB",
        "params": "1B",
    },
    {
        "id": "unsloth/Llama-3.2-3B-Instruct",
        "name": "Llama-3.2 3B",
        "description": "Better quality responses",
        "vram": "~8GB",
        "params": "3B",
    },
    {
        "id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "name": "Llama-3.1 8B",
        "description": "High quality conversations",
        "vram": "~16GB",
        "params": "8B",
    },
    {
        "id": "unsloth/Qwen2.5-7B-Instruct",
        "name": "Qwen-2.5 7B",
        "description": "Excellent reasoning capabilities",
        "vram": "~14GB",
        "params": "7B",
    },
    {
        "id": "unsloth/Phi-3.5-mini-instruct",
        "name": "Phi-3.5 Mini",
        "description": "Microsoft's compact model",
        "vram": "~8GB",
        "params": "3.8B",
    },
    {
        "id": "unsloth/mistral-7b-instruct-v0.3",
        "name": "Mistral 7B v0.3",
        "description": "Strong general purpose model",
        "vram": "~14GB",
        "params": "7B",
    },
    {
        "id": "unsloth/Ministral-8B-Instruct-2410",
        "name": "Ministral 8B",
        "description": "Latest Ministral release",
        "vram": "~16GB",
        "params": "8B",
    },
    {
        "id": "unsloth/Llama-3.3-70B-Instruct",
        "name": "Llama-3.3 70B",
        "description": "Best quality (requires large GPU)",
        "vram": "~40GB",
        "params": "70B",
    },
]

DEFAULT_MODEL = "unsloth/gemma-3-270m-it"


def get_model_choices():
    """Get list of model choices for dropdown"""
    return [model["id"] for model in AVAILABLE_MODELS]


def get_default_model():
    """Get default model"""
    return DEFAULT_MODEL


def get_model_info(model_id: str) -> dict:
    """Get detailed information about a model"""
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model
    
    # Return unknown model info if not found
    return {
        "id": model_id,
        "name": model_id.split("/")[-1],
        "description": "Custom model",
        "vram": "Unknown",
        "params": "Unknown",
    }

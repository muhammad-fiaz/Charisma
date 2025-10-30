"""Data processor for preparing training data from Notion memories"""

from datetime import datetime
from typing import Dict, List, Optional

try:
    from datasets import Dataset
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.utils.logger import get_logger

logger = get_logger()


class DataProcessor:
    """Processes user data and memories into training format"""

    def __init__(self, system_prompt_template: Optional[str] = None, response_structure: Optional[str] = None):
        """
        Initialize DataProcessor
        
        Args:
            system_prompt_template: Custom system prompt template with {placeholders}
            response_structure: Custom response structure guidance
        """
        self.system_prompt_template = system_prompt_template
        self.response_structure = response_structure

    def create_training_dataset(
        self, personal_info: Dict[str, str], memories: List[Dict], selected_memory_ids: Optional[List[str]] = None
    ) -> Dataset:
        """
        Create training dataset from personal info and memories
        
        Args:
            personal_info: Dictionary of personal information
            memories: List of all cached memories
            selected_memory_ids: Optional list of memory IDs to include. If None, uses all memories.
        
        Returns:
            Dataset ready for training
        """
        # Filter memories based on selection
        if selected_memory_ids:
            filtered_memories = [m for m in memories if m.get("id") in selected_memory_ids]
            logger.info(f"Using {len(filtered_memories)} selected memories out of {len(memories)} total")
        else:
            filtered_memories = memories
            logger.info(f"Using all {len(memories)} memories")
        
        system_prompt = self._create_system_prompt(personal_info)
        conversations = []

        for memory in filtered_memories:
            conversation = self._memory_to_conversation(memory, system_prompt)
            if conversation:  # Only add valid conversations
                conversations.append(conversation)

        dataset = Dataset.from_dict({"conversations": conversations})
        logger.info(f"Created training dataset with {len(conversations)} examples")
        return dataset

    def _create_system_prompt(self, personal_info: Dict[str, str]) -> str:
        """Create system prompt from personal information"""
        
        # If custom template is provided, use it
        if self.system_prompt_template:
            try:
                # Create a safe dict with defaults for missing values
                format_dict = {
                    "name": personal_info.get("name", "the user"),
                    "age": personal_info.get("age", ""),
                    "country": personal_info.get("country", ""),
                    "location": personal_info.get("location", ""),
                    "hobbies": personal_info.get("hobbies", ""),
                    "favorites": personal_info.get("favorites", ""),
                }
                
                # Add any additional fields from personal_info
                for key, value in personal_info.items():
                    if key not in format_dict:
                        format_dict[key] = value
                
                prompt = self.system_prompt_template.format(**format_dict)
                
                # Append response structure if provided
                if self.response_structure:
                    prompt += f"\n\n{self.response_structure}"
                
                return prompt
            except KeyError as e:
                logger.warning(f"Missing placeholder in system prompt template: {e}. Using default.")
        
        # Default system prompt (fallback)
        prompt_parts = ["You are an AI representation of the following person:\n"]

        if "name" in personal_info and personal_info["name"]:
            prompt_parts.append(f"Name: {personal_info['name']}")
        if "age" in personal_info and personal_info["age"]:
            prompt_parts.append(f"Age: {personal_info['age']}")
        if "country" in personal_info and personal_info["country"]:
            prompt_parts.append(f"Country: {personal_info['country']}")
        if "location" in personal_info and personal_info["location"]:
            prompt_parts.append(f"Location: {personal_info['location']}")
        if "hobbies" in personal_info and personal_info["hobbies"]:
            prompt_parts.append(f"Hobbies: {personal_info['hobbies']}")
        if "favorites" in personal_info and personal_info["favorites"]:
            prompt_parts.append(f"Favorites: {personal_info['favorites']}")

        excluded = {"name", "age", "country", "location", "hobbies", "favorites"}
        for key, value in personal_info.items():
            if key not in excluded and value:
                prompt_parts.append(f"{key.capitalize()}: {value}")

        prompt_parts.append(
            "\nRespond as this person would, using their memories, experiences, and personality."
        )
        
        if self.response_structure:
            prompt_parts.append(f"\n{self.response_structure}")
        
        return "\n".join(prompt_parts)

    def _memory_to_conversation(
        self, memory: Dict, system_prompt: str
    ) -> List[Dict[str, str]]:
        """Convert a memory into conversation format"""
        content = memory.get("content", "")
        date = memory.get("date", "")
        title = memory.get("title", "Memory")

        date_str = ""
        if date:
            try:
                if isinstance(date, datetime):
                    date_str = date.strftime("%B %d, %Y")
                else:
                    date_obj = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
                    date_str = date_obj.strftime("%B %d, %Y")
            except Exception:
                pass  # If date parsing fails, use title instead

        if date_str:
            user_prompt = f"Tell me about what happened on {date_str}."
        else:
            user_prompt = f"Tell me about: {title}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": content},
        ]

    def format_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Apply chat template to dataset"""

        def formatting_func(examples):
            convos = examples["conversations"]
            texts = [
                tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                ).removeprefix("<bos>")
                for convo in convos
            ]
            return {"text": texts}

        dataset = dataset.map(formatting_func, batched=True)
        logger.info("Applied chat template to dataset")
        return dataset

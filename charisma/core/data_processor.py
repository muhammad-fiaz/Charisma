"""Data processor for preparing training data from Notion memories"""

import platform
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
        
        # If custom template is provided from config/settings, use it
        if self.system_prompt_template:
            try:
                # Create a safe dict with defaults for missing values
                format_dict = {
                    "name": personal_info.get("name", "the user"),
                    "age": personal_info.get("age", ""),
                    "gender": personal_info.get("gender", ""),
                    "country": personal_info.get("country", ""),
                    "location": personal_info.get("location", ""),
                    "hobbies": personal_info.get("hobbies", ""),
                    "favorites": personal_info.get("favorites", ""),
                    "bio": personal_info.get("bio", ""),
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
                logger.error(f"Missing placeholder in system prompt template: {e}. Please check your Settings.")
                raise ValueError(f"Invalid system prompt template - missing placeholder: {e}")
        
        # If no template provided, raise error - must be configured in Settings
        raise ValueError(
            "No system prompt template configured. Please configure prompts in Settings tab or charisma.toml"
        )

    def _memory_to_conversation(
        self, memory: Dict, system_prompt: str
    ) -> List[Dict[str, str]]:
        """Convert a memory into conversation format"""
        # Support both 'content' and '_content' fields from Notion
        content = memory.get("content") or memory.get("_content", "")
        date = memory.get("date") or memory.get("created_time", "")
        
        # Extract title from properties or use default
        title = "Memory"
        if "properties" in memory and "title" in memory["properties"]:
            title_prop = memory["properties"]["title"]
            if title_prop.get("type") == "title" and title_prop.get("title"):
                title = title_prop["title"][0]["plain_text"]
        elif "title" in memory:
            title = memory["title"]

        if not content or not content.strip():
            return None

        # Create diverse, natural conversation starters based on the memory
        import random
        
        # Generate contextual prompts that encourage persona-based responses
        prompts = [
            # Identity questions
            "Who are you?",
            "Tell me about yourself",
            "What's your name?",
            "Introduce yourself",
            
            # Memory-based questions
            f"What do you remember about {title.lower()}?",
            f"Tell me about {title.lower()}",
            f"Can you share your experience with {title.lower()}?",
            
            # Open-ended prompts
            "What have you been up to lately?",
            "What's on your mind?",
            "How are you feeling today?",
            "What did you do recently?",
            
            # Simple greetings that should get persona responses
            "Hi!",
            "Hey, how's it going?",
            "What's up?",
        ]
        
        # Add time-based prompt if date is available
        if date:
            try:
                if isinstance(date, datetime):
                    date_str = date.strftime("%B %d, %Y")
                else:
                    date_obj = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
                    date_str = date_obj.strftime("%B %d, %Y")
                
                prompts.append(f"What did you do on {date_str}?")
                prompts.append(f"Tell me about {date_str}")
            except Exception:
                pass
        
        # Select a random prompt to create variety in training
        user_prompt = random.choice(prompts)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": content},
        ]

    def format_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Apply chat template to dataset and tokenize for SFTTrainer"""

        def formatting_func(examples):
            convos = examples["conversations"]
            texts = [
                tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                ).removeprefix("<bos>")
                for convo in convos
            ]
            return {"text": texts}

        # Windows requires num_proc=1 to avoid pickling issues
        num_proc = 1 if platform.system() == "Windows" else None
        dataset = dataset.map(formatting_func, batched=True, num_proc=num_proc)

        logger.info("Applied chat template to dataset")
        return dataset

"""Main Gradio application"""

from typing import Dict, List

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.config import ConfigManager, AVAILABLE_MODELS, DEFAULT_MODEL
from charisma.integrations import NotionClient, HuggingFaceClient
from charisma.core import DataProcessor, ModelManager, Trainer
from charisma.ui.tabs import (
    create_main_tab, 
    create_settings_tab, 
    create_logs_tab,
    create_inference_tab,
    create_huggingface_tab
)
from charisma.utils.logger import get_logger, setup_logger
from charisma.utils import CacheManager

setup_logger()
logger = get_logger()


class CharismaApp:
    """Main Charisma application"""

    def __init__(self, config_path: str = "charisma.toml"):
        self.config_manager = ConfigManager(config_path)
        self.notion_client = None
        self.hf_client = None
        self.current_memories = []
        self.memory_display_map = {}
        self.model_manager = None
        self.trainer = None
        self.cache_manager = CacheManager()  # Initialize cache manager
        self.inference_model = None  # Model for inference
        self.inference_tokenizer = None  # Tokenizer for inference
        self.training_stopped = False  # Flag to stop training
        self.inference_stopped = False  # Flag to stop inference
        self.training_active = False  # Flag to track if training is running
        self.inference_active = False  # Flag to track if inference is running

        logger.info("Charisma application initialized")
        
        # Auto-load cached memories on startup
        self._load_cached_memories_on_startup()
    
    def _load_cached_memories_on_startup(self):
        """Load cached memories automatically on startup"""
        try:
            cached_memories = self.cache_manager.load_memories()
            if cached_memories:
                self.current_memories = cached_memories
                logger.info(f"Loaded {len(cached_memories)} cached memories from local storage")
                
                # Create memory display map
                if self.notion_client is None:
                    # Create a temporary client just for formatting
                    temp_client = NotionClient()
                    memory_choices = temp_client.format_pages_for_display(cached_memories)
                else:
                    memory_choices = self.notion_client.format_pages_for_display(cached_memories)
                
                self.memory_display_map = {}
                for i, page in enumerate(cached_memories):
                    if i < len(memory_choices):
                        self.memory_display_map[memory_choices[i]] = page
                
                logger.success(f"Auto-loaded {len(cached_memories)} memories from cache")
            else:
                logger.info("No cached memories found - will need to connect to Notion")
        except Exception as e:
            logger.warning(f"Could not auto-load cached memories: {e}")

    def get_cached_memory_info(self) -> Dict:
        """Get information about cached memories"""
        cache_stats = self.cache_manager.get_cache_stats()
        has_cache = cache_stats["total_memories"] > 0
        
        return {
            "has_cache": has_cache,
            "total_memories": cache_stats["total_memories"],
            "cache_size_mb": cache_stats["cache_size_mb"],
            "last_updated": cache_stats["last_updated"],
            "cache_dir": cache_stats["cache_dir"]
        }
    
    def get_memory_choices(self) -> list:
        """Get list of memory choice strings for UI display"""
        if not self.current_memories:
            return []
        
        # Use NotionClient to format for display
        if not self.notion_client:
            temp_client = NotionClient()
            return temp_client.format_pages_for_display(self.current_memories)
        else:
            return self.notion_client.format_pages_for_display(self.current_memories)

    def connect_notion(self) -> Dict:
        """Connect to Notion and fetch memories using browser OAuth"""
        try:
            # Check for OAuth credentials first
            oauth_client_id = self.config_manager.get("notion", "oauth_client_id", "")
            oauth_client_secret = self.config_manager.get("notion", "oauth_client_secret", "")
            api_key = self.config_manager.get("notion", "api_key", "")
            
            # Prefer OAuth over API key
            if oauth_client_id and oauth_client_secret:
                logger.info("Attempting Notion OAuth authentication...")
                
                # Initialize client with OAuth credentials
                self.notion_client = NotionClient(
                    client_id=oauth_client_id,
                    client_secret=oauth_client_secret
                )
                
                # Open browser for authentication
                logger.info("🌐 Opening browser for Notion authentication...")
                if not self.notion_client.authenticate_with_browser():
                    return {
                        "success": False,
                        "error": "Browser authentication failed or was cancelled.",
                    }
                
                logger.success("✅ Notion OAuth authentication successful!")
                
                # Store OAuth data in config for future use
                oauth_data = self.notion_client.get_oauth_data()
                if oauth_data:
                    self.config_manager.set("notion", "access_token", oauth_data.get("access_token", ""))
                    self.config_manager.set("notion", "refresh_token", oauth_data.get("refresh_token", ""))
                    self.config_manager.set("notion", "workspace_name", oauth_data.get("workspace_name", ""))
                    self.config_manager.set("notion", "workspace_id", oauth_data.get("workspace_id", ""))
                    self.config_manager.set("notion", "bot_id", oauth_data.get("bot_id", ""))
                    self.config_manager.save_config()
                    logger.info("💾 OAuth credentials saved to config")
                
            elif api_key:
                # Fallback to API key authentication
                logger.info("Using Notion API key authentication...")
                self.notion_client = NotionClient(api_key)
                
                if not self.notion_client.test_connection():
                    return {
                        "success": False,
                        "error": "Failed to connect to Notion. Check your API key.",
                    }
            else:
                return {
                    "success": False,
                    "error": "No Notion credentials configured. Please add OAuth credentials or API key in Settings.\n\n"
                            "To use OAuth:\n"
                            "1. Go to https://www.notion.so/my-integrations\n"
                            "2. Create a new integration\n"
                            "3. Set 'Redirect URIs' to: http://localhost:8888/callback\n"
                            "4. Copy the Client ID and Client Secret to Settings",
                }

            # Fetch all accessible content (pages and databases)
            logger.info("🔍 Fetching accessible Notion content...")
            content = self.notion_client.get_all_content()
            pages = content.get("pages", [])
            databases = content.get("databases", [])
            
            # Cache memories locally for faster training
            logger.info("💾 Caching memories locally...")
            self.cache_manager.save_memories(pages)
            cache_stats = self.cache_manager.get_cache_stats()
            logger.success(f"✅ Cached {cache_stats['total_memories']} memories ({cache_stats['cache_size_mb']} MB)")
            
            # Store full page data
            self.current_memories = pages
            
            # Get formatted display strings
            memory_choices = self.notion_client.format_pages_for_display(pages)
            
            # Create mapping from display string to page data for later lookup
            self.memory_display_map = {}
            for i, page in enumerate(pages):
                if i < len(memory_choices):
                    self.memory_display_map[memory_choices[i]] = page

            total_items = len(pages) + len(databases)
            logger.success(f"✅ Connected to Notion. Found {len(pages)} pages and {len(databases)} databases")
            
            # Create formatted summary
            summary_parts = []
            if self.notion_client.workspace_name:
                summary_parts.append(f"**Workspace:** {self.notion_client.workspace_name}")
            summary_parts.append(f"**Pages:** {len(pages)}")
            summary_parts.append(f"**Databases:** {len(databases)}")
            summary_parts.append(f"**Total Items:** {total_items}")
            
            success_summary = "\n".join(summary_parts)
            
            return {
                "success": True,
                "memories": pages,
                "memory_choices": memory_choices,
                "databases": databases,
                "total_items": total_items,
                "summary": success_summary,
            }

        except Exception as e:
            logger.error(f"❌ Failed to connect to Notion: {e}")
            return {"success": False, "error": str(e)}

    def test_notion_connection(self, api_key: str) -> Dict:
        """Test Notion connection with provided API key"""
        try:
            client = NotionClient(api_key)
            if client.test_connection():
                pages = client.get_all_pages()
                return {"success": True, "pages_count": len(pages)}
            else:
                return {"success": False, "error": "Authentication failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_hf_connection(self, token: str) -> Dict:
        """Test HuggingFace connection with provided token"""
        try:
            client = HuggingFaceClient()
            client.login(token)
            if client.test_connection():
                # Get username from token
                from huggingface_hub import whoami

                info = whoami(token)
                return {"success": True, "username": info["name"]}
            else:
                return {"success": False, "error": "Authentication failed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_config(self, updates: Dict) -> Dict:
        """Save configuration updates"""
        try:
            for key, value in updates.items():
                # Split dot notation into section and key
                if "." in key:
                    section, config_key = key.split(".", 1)
                    self.config_manager.set(section, config_key, value)
                else:
                    # If no dot, treat as section-level config
                    logger.warning(f"Config key without section: {key}")
            self.config_manager.save_config()
            logger.info("Configuration saved")
            return {"success": True}
        except Exception as e:
            logger.error("Failed to save config: %s", str(e))
            return {"success": False, "error": str(e)}

    def generate_clone(
        self,
        personal_info: Dict[str, str],
        selected_memories: List[str],
        model_name: str,
        use_full_finetune: bool,
        output_name: str,
    ) -> Dict:
        """Generate AI clone from memories"""
        try:
            logger.info("Starting AI clone generation")
            self.training_active = True  # Mark training as active
            logger.info("Personal info: %s", personal_info)
            logger.info("Model: %s", model_name)
            logger.info(
                "Training mode: %s", "Full fine-tune" if use_full_finetune else "LoRA"
            )

            # Filter selected memories
            if not selected_memories:
                return {"success": False, "error": "No memories selected"}

            # Get selected memory IDs from display strings
            selected_memory_ids = []
            for display_str in selected_memories:
                if display_str in self.memory_display_map:
                    page = self.memory_display_map[display_str]
                    selected_memory_ids.append(page.get("id"))
                else:
                    logger.warning(f"Could not find page data for: {display_str}")

            if not selected_memory_ids:
                return {"success": False, "error": "Could not find matching memory data for selections"}

            logger.info(f"Selected {len(selected_memory_ids)} memories for training")

            # Load memories from cache (faster than API calls)
            logger.info("📂 Loading memories from local cache...")
            cached_memories = self.cache_manager.load_memories(selected_memory_ids)
            
            if not cached_memories:
                logger.warning("No cached memories found, using in-memory data")
                # Fallback to in-memory data
                cached_memories = [
                    self.memory_display_map[display_str] 
                    for display_str in selected_memories 
                    if display_str in self.memory_display_map
                ]
            
            logger.success(f"✅ Loaded {len(cached_memories)} memories from cache")

            # Get prompt configuration from config (Settings tab)
            system_prompt_template = self.config_manager.get("prompts", "system_prompt", "")
            response_structure = self.config_manager.get("prompts", "response_structure", "")

            # Prepare dataset with custom prompts
            logger.info("🔨 Preparing training dataset...")
            data_processor = DataProcessor(
                system_prompt_template=system_prompt_template,
                response_structure=response_structure
            )
            dataset = data_processor.create_training_dataset(
                personal_info, cached_memories, selected_memory_ids
            )

            # Load model
            model_config = self.config_manager.get_section("model")
            unsloth_config = self.config_manager.get_section("unsloth")
            lora_config = (
                self.config_manager.get_section("lora")
                if not use_full_finetune
                else None
            )
            
            # Merge unsloth config into lora_config for LoRA fine-tuning
            if lora_config and unsloth_config:
                lora_config.update({
                    "use_gradient_checkpointing": unsloth_config.get("use_gradient_checkpointing", "unsloth"),
                    "use_rslora": unsloth_config.get("use_rslora", False),
                    "use_loftq": unsloth_config.get("use_loftq", False),
                })

            self.model_manager = ModelManager()
            model, tokenizer = self.model_manager.load_model(
                model_name=model_name,
                max_seq_length=model_config.get("max_seq_length", 2048),
                load_in_4bit=model_config.get("load_in_4bit", True),
                use_lora=not use_full_finetune,
                lora_config=lora_config,
            )

            # Format dataset
            formatted_dataset = data_processor.format_dataset(dataset, tokenizer)

            # Train model - merge training and unsloth configs
            training_config = self.config_manager.get_section("training")
            training_config.update(unsloth_config)  # Add unsloth settings to training config
            output_dir = f"./outputs/{output_name}"

            self.trainer = Trainer(model, tokenizer)

            if use_full_finetune:
                self.trainer.full_finetune(
                    formatted_dataset, training_config, output_dir
                )
            else:
                self.trainer.train(formatted_dataset, training_config, output_dir)

            # Save model
            save_method = "lora" if not use_full_finetune else "merged_16bit"
            self.model_manager.save_model(output_dir, save_method=save_method)

            # Get stats
            stats = self.trainer.get_training_stats()

            logger.info("AI clone generation completed successfully")
            self.training_active = False  # Mark training as inactive
            return {
                "success": True,
                "output_dir": output_dir,
                "stats": {
                    "total_memories": len(cached_memories),
                    "steps": stats.get("current_step", 0),
                    "final_loss": stats.get("loss", 0),
                },
            }

        except Exception as e:
            logger.error("Failed to generate clone: %s", str(e), exc_info=True)
            self.training_active = False  # Mark training as inactive on error
            return {"success": False, "error": str(e)}

    def stop_training(self):
        """Stop the training process"""
        logger.info("Stop training requested")
        self.training_stopped = True
        self.training_active = False  # Mark training as inactive
        if self.trainer:
            # Signal trainer to stop (implementation depends on trainer)
            logger.info("Signaling trainer to stop...")

    def load_model_for_inference(self, model_path: str) -> Dict:
        """Load a fine-tuned model for inference"""
        try:
            from pathlib import Path
            
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Model path does not exist: {model_path}"
                }
            
            logger.info(f"Loading model for inference from: {model_path}")
            
            # Initialize model manager if needed
            if not self.model_manager:
                self.model_manager = ModelManager()
            
            # Load the model and tokenizer
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.inference_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.inference_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            logger.success(f"✅ Model loaded successfully from {model_path}")
            
            return {
                "success": True,
                "model_type": self.inference_model.__class__.__name__,
                "model_path": str(model_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def run_inference(
        self, 
        message: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.1,
        system_prompt: str = None
    ) -> Dict:
        """Run inference on the loaded model"""
        try:
            if not self.inference_model or not self.inference_tokenizer:
                return {
                    "success": False,
                    "error": "No model loaded. Please load a model first."
                }
            
            self.inference_active = True  # Mark inference as active
            
            self.inference_stopped = False
            
            # Build conversation
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})
            
            # Apply chat template
            formatted_prompt = self.inference_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.inference_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.inference_model.device)
            
            # Generate
            import torch
            with torch.no_grad():
                outputs = self.inference_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.inference_tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.inference_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"Generated response: {response[:100]}...")
            
            self.inference_active = False  # Mark inference as inactive
            return {
                "success": True,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            self.inference_active = False  # Mark inference as inactive on error
            return {"success": False, "error": str(e)}

    def stop_inference(self):
        """Stop the inference process"""
        logger.info("Stop inference requested")
        self.inference_stopped = True
        self.inference_active = False  # Mark inference as inactive

    def push_to_huggingface(
        self,
        model_path: str,
        repo_id: str,
        private: bool = True,
        create_repo: bool = True,
        description: str = "",
        tags: List[str] = None,
        commit_message: str = "Upload model",
        include_training_args: bool = True,
        include_tokenizer: bool = True,
    ):
        """Push model to Hugging Face Hub (generator for progress updates)"""
        try:
            from pathlib import Path
            
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                yield {
                    "error": f"Model path does not exist: {model_path}"
                }
                return
            
            yield {
                "status": "Initializing Hugging Face client...",
                "message": "Connecting to Hugging Face Hub..."
            }
            
            # Initialize HF client
            hf_token = self.config_manager.get("huggingface", "token", "")
            if not hf_token:
                yield {
                    "error": "No Hugging Face token found. Please add your token in Settings."
                }
                return
            
            if not self.hf_client:
                self.hf_client = HuggingFaceClient()
            
            self.hf_client.login(hf_token)
            
            yield {
                "status": "Preparing repository...",
                "message": f"Repository: {repo_id}\nPrivate: {private}"
            }
            
            # Create repository if needed
            from huggingface_hub import HfApi, create_repo
            api = HfApi()
            
            if create_repo:
                try:
                    create_repo(
                        repo_id=repo_id,
                        private=private,
                        exist_ok=True,
                        token=hf_token
                    )
                    yield {
                        "status": "Repository ready",
                        "message": f"Repository {repo_id} is ready"
                    }
                except Exception as e:
                    logger.warning(f"Could not create repo: {e}")
            
            yield {
                "status": "Uploading model files...",
                "message": "This may take a few minutes depending on model size..."
            }
            
            # Upload the model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load and push model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.push_to_hub(
                repo_id=repo_id,
                private=private,
                commit_message=commit_message,
                token=hf_token
            )
            
            yield {
                "status": "Model uploaded successfully",
                "message": "Model files uploaded to Hugging Face Hub"
            }
            
            # Upload tokenizer if requested
            if include_tokenizer:
                yield {
                    "status": "Uploading tokenizer...",
                    "message": "Uploading tokenizer files..."
                }
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                tokenizer.push_to_hub(
                    repo_id=repo_id,
                    private=private,
                    commit_message=commit_message,
                    token=hf_token
                )
            
            # Create model card
            yield {
                "status": "Creating model card...",
                "message": "Generating README.md..."
            }
            
            model_card = f"""---
tags:
{chr(10).join(f'- {tag}' for tag in (tags or []))}
license: mit
---

# {repo_id.split('/')[-1]}

{description if description else 'A fine-tuned AI model created with Charisma.'}

## Model Details

- **Created with:** Charisma - Personal Memory Clone
- **Base Model:** Fine-tuned language model
- **Training:** LoRA fine-tuning on personal memories

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "user", "content": "Hello!"}}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training

This model was fine-tuned using personal memories to create an AI clone that mimics personality and knowledge.

"""
            
            # Upload model card
            api.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=hf_token,
                commit_message="Add model card"
            )
            
            repo_url = f"https://huggingface.co/{repo_id}"
            
            yield {
                "success": True,
                "repo_url": repo_url,
                "message": f"✅ Successfully uploaded to {repo_url}"
            }
            
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}", exc_info=True)
            yield {
                "error": str(e)
            }

    def create_ui(self, share: bool = False) -> gr.Blocks:
        """Create Gradio UI"""
        
        # Custom CSS for better UI - Responsive and Centered
        custom_css = """
      
        
        /* Responsive groups and rows */
        .gr-group {
            margin-bottom: 20px !important;
            padding: 20px !important;
            border-radius: 12px !important;
        }
        
        .row {
            margin-bottom: 15px !important;
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 15px !important;
        }
        
        /* Buttons - full width and proper spacing */
        .gr-button {
            width: 100% !important;
            margin: 10px 0 !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
        }
        
        /* Input fields responsive */
        .gr-textbox, .gr-number, .gr-dropdown {
            width: 100% !important;
        }
        
        /* Scrollable checkbox group */
        .scrollable-checkboxgroup {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        }
        .scrollable-checkboxgroup label {
            display: block;
            padding: 8px;
            margin: 4px 0;
        }
        
        /* Console styling */
        #training-console {
            font-family: 'Courier New', monospace !important;
            background: #1e1e1e !important;
            color: #d4d4d4 !important;
            border-radius: 8px !important;
            width: 100% !important;
        }
        
        /* Section headers */
        .section-header {
            margin-top: 20px !important;
            margin-bottom: 15px !important;
            text-align: center !important;
        }
        
      
        
        /* Notion loading indicator */
        #notion-loading {
            text-align: center;
            padding: 15px;
            background: #f0f7ff;
            border-radius: 8px;
            margin: 10px 0;
        }
```
        .subsection-header {
            margin-top: 15px !important;
            margin-bottom: 10px !important;
        }
        /* Add margin between rows */
        .row {
            margin-bottom: 15px !important;
        }
        /* Add margin between groups */
        .gr-group {
            margin-bottom: 20px !important;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner {
            display: inline-block;
            animation: spin 1s linear infinite;
        }
        """
        
        with gr.Blocks(
            title="Charisma - Personal Memory Clone", 
            theme=gr.themes.Soft(),
            css=custom_css
        ) as app:
            gr.Markdown(
                """
                <div style="text-align: center;">
                    <h1>🧠 Charisma</h1>
                    <p>Clone your memory and personality using AI</p>
                    <p style="font-size: 0.9em; color: #666;">
                        🔒 Fully local processing - No data is collected on our servers
                    </p>
                </div>
                """
            )

            with gr.Tabs():
                with gr.Tab("Main"):
                    create_main_tab(
                        on_connect_notion=self.connect_notion,
                        on_select_memories=lambda selected: {
                            "success": True,
                            "choices": self.get_memory_choices()
                        },
                        on_generate=self.generate_clone,
                        on_stop_training=self.stop_training,
                        on_get_cached_memories=self.get_cached_memory_info,
                        available_models=AVAILABLE_MODELS,
                        default_model=DEFAULT_MODEL,
                        training_active=self.training_active,
                    )

                with gr.Tab("Inference"):
                    create_inference_tab(
                        on_load_model=self.load_model_for_inference,
                        on_inference=self.run_inference,
                        on_stop_inference=self.stop_inference,
                        inference_active=self.inference_active,
                    )

                with gr.Tab("Push to Hub"):
                    create_huggingface_tab(
                        config_manager=self.config_manager,
                        on_push_to_hub=self.push_to_huggingface,
                    )

                with gr.Tab("Settings"):
                    create_settings_tab(
                        config_manager=self.config_manager,
                        on_save_config=self.save_config,
                        on_test_notion=self.test_notion_connection,
                        on_test_hf=self.test_hf_connection,
                    )

                with gr.Tab("Logs"):
                    create_logs_tab(log_dir="./logs")

            gr.Markdown(
                """
                <div style="text-align: center; margin-top: 20px; font-size: 0.8em; color: #999;">
                    <p>Charisma v0.1.0 | Built with ❤️ using Unsloth & Gradio</p>
                    <p>⚠️ PRIVACY NOTICE: All processing happens locally. Your data never leaves your machine.</p>
                </div>
                """
            )

        return app

    def launch(self, share: bool = False, **kwargs):
        """Launch the Gradio application"""
        logger.info("Launching Charisma UI (share=%s)", share)
        app = self.create_ui(share=share)
        app.launch(share=share, **kwargs)

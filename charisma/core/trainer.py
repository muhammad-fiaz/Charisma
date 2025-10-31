"""Trainer for fine-tuning models"""

import os
import platform
from typing import Dict, Optional, Callable

# Force single process on Windows to avoid pickling issues with UnslothSFTTrainer
# Windows multiprocessing uses 'spawn' instead of 'fork', which requires pickling
# UnslothSFTTrainer can't be pickled, so we must use num_proc=1 on Windows
if platform.system() == "Windows":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported, FastLanguageModel
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.utils.logger import get_logger

logger = get_logger()


class Trainer:
    """Handles model training with Unsloth and TRL"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = None

    def train(
        self,
        dataset: Dataset,
        training_config: Dict,
        output_dir: str = "./charisma_output",
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Train the model on prepared dataset"""
        logger.info("Starting training with %d examples", len(dataset))

        # Prepare training arguments
        args = TrainingArguments(
            per_device_train_batch_size=training_config.get("batch_size", 2),
            gradient_accumulation_steps=training_config.get(
                "gradient_accumulation_steps", 4
            ),
            warmup_steps=training_config.get("warmup_steps", 5),
            num_train_epochs=training_config.get("num_epochs", 1),
            max_steps=training_config.get("max_steps", 60),
            learning_rate=float(training_config.get("learning_rate", 2e-4)),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=training_config.get("logging_steps", 1),
            optim=training_config.get("optimizer", "adamw_8bit"),
            weight_decay=training_config.get("weight_decay", 0.01),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
            seed=3407,
            output_dir=output_dir,
            report_to="none",  # Disable wandb/tensorboard
            dataloader_num_workers=0 if platform.system() == "Windows" else 2,  # Force single-thread on Windows
        )

        # Ensure single-process mode on Windows
        if platform.system() == "Windows":
            # Force environment variable to ensure single-process mode
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Create trainer with dataset_text_field to use pre-formatted text
        # This prevents SFTTrainer from doing its own multiprocess tokenization
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",  # Use the pre-formatted "text" field directly
            max_seq_length=training_config.get("max_seq_length", 2048),
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            packing=training_config.get("packing", False),
            args=args,
            dataset_num_proc=1,  # Force single process for dataset operations on all platforms
        )
        
        # Use Unsloth's train_on_responses_only to mask instruction parts
        # This improves accuracy by only training on the assistant's responses
        try:
            from unsloth.chat_templates import train_on_responses_only
            self.trainer = train_on_responses_only(
                self.trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )
            logger.info("✅ Enabled train_on_responses_only - will only train on assistant outputs")
        except Exception as e:
            logger.warning(f"Could not enable train_on_responses_only: {e}. Continuing without masking.")

        # Train
        logger.info("Training started...")
        self.trainer.train()
        logger.info("Training completed!")

    def full_finetune(
        self,
        dataset: Dataset,
        training_config: Dict,
        output_dir: str = "./charisma_output",
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Full fine-tuning (all parameters)"""
        logger.info("Starting FULL fine-tuning with %d examples", len(dataset))

        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        logger.info("All model parameters unfrozen for full fine-tuning")

        # Prepare training arguments (more conservative for full fine-tune)
        args = TrainingArguments(
            per_device_train_batch_size=training_config.get(
                "batch_size", 1
            ),  # Smaller batch
            gradient_accumulation_steps=training_config.get(
                "gradient_accumulation_steps", 8
            ),  # More accumulation
            warmup_steps=training_config.get("warmup_steps", 10),
            num_train_epochs=training_config.get("num_epochs", 1),
            max_steps=training_config.get("max_steps", 60),
            learning_rate=float(training_config.get("learning_rate", 5e-5)),  # Lower LR
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=training_config.get("logging_steps", 1),
            optim=training_config.get("optimizer", "adamw_8bit"),
            weight_decay=training_config.get("weight_decay", 0.01),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            dataloader_num_workers=0 if platform.system() == "Windows" else 2,  # Force single-thread on Windows
        )

        # Ensure single-process mode on Windows
        if platform.system() == "Windows":
            # Force environment variable to ensure single-process mode
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Create trainer with dataset_text_field to use pre-formatted text
        # This prevents SFTTrainer from doing its own multiprocess tokenization
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",  # Use the pre-formatted "text" field directly
            max_seq_length=training_config.get("max_seq_length", 2048),
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            packing=training_config.get("packing", False),
            args=args,
            dataset_num_proc=1,  # Force single process for dataset operations on all platforms
        )
        
        # Use Unsloth's train_on_responses_only to mask instruction parts
        # This improves accuracy by only training on the assistant's responses
        try:
            from unsloth.chat_templates import train_on_responses_only
            self.trainer = train_on_responses_only(
                self.trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )
            logger.info("✅ Enabled train_on_responses_only - will only train on assistant outputs")
        except Exception as e:
            logger.warning(f"Could not enable train_on_responses_only: {e}. Continuing without masking.")

        # Train
        logger.info("Full fine-tuning started...")
        self.trainer.train()
        logger.info("Full fine-tuning completed!")

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        if not self.trainer or not self.trainer.state:
            return {}

        state = self.trainer.state
        return {
            "current_step": state.global_step,
            "max_steps": state.max_steps,
            "current_epoch": state.epoch,
            "total_epochs": state.num_train_epochs,
            "loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
            "learning_rate": state.log_history[-1].get("learning_rate", 0)
            if state.log_history
            else 0,
        }

    def test_inference(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Test inference with a prompt"""
        FastLanguageModel.for_inference(self.model)

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            top_p=0.9,
        )

        response = self.tokenizer.batch_decode(outputs)[0]

        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[
                -1
            ]
            response = response.split("<|eot_id|>")[0].strip()

        return response

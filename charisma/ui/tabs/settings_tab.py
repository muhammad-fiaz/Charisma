"""Settings tab for Gradio UI"""

from typing import Callable

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.utils.logger import get_logger

logger = get_logger()


def create_settings_tab(
    config_manager,
    on_save_config: Callable,
    on_test_notion: Callable,
    on_test_hf: Callable,
) -> gr.Column:
    """Create the settings tab"""

    with gr.Column() as settings_tab:
        gr.Markdown("# ⚙️ Settings")
        gr.Markdown("Configure Charisma settings. Changes are saved to `charisma.toml`")

        # API Keys Section
        with gr.Group():
            gr.Markdown("### 🔑 API Credentials", elem_classes="section-header")
            
            # Notion Internal Integration (API Key)
            gr.Markdown("#### Notion Integration", elem_classes="subsection-header")
            gr.Markdown(
                "**Setup Instructions:**\n"
                "1. Go to [Notion Integrations](https://www.notion.so/profile/integrations)\n"
                "2. Login to your Notion account\n"
                "3. Click '+ New integration'\n"
                "4. Set type to **'Internal'** (keeps it private)\n"
                "5. Enable **'Read content'** capability\n"
                "6. Copy the 'Internal Integration Secret' below\n"
                "7. Share your memory pages with the integration"
            )

            with gr.Row():
                notion_token = gr.Textbox(
                    label="Notion Internal Integration Secret",
                    type="password",
                    value=config_manager.get("notion", "api_key", ""),
                    placeholder="secret_xxxxxxxxxxxxx",
                    info="Your Internal Integration Secret from Notion",
                )
            
            with gr.Row():
                test_notion_btn = gr.Button(
                    "🧪 Test Connection", variant="secondary"
                )

            notion_test_result = gr.Textbox(
                label="Test Result", interactive=False
            )
            
            gr.Markdown("---")
            
            gr.Markdown("#### HuggingFace", elem_classes="subsection-header")

            with gr.Row():
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    type="password",
                    value=config_manager.get("huggingface", "token", ""),
                    placeholder="hf_xxxxxxxxxxxxx",
                    info="Get your token from https://huggingface.co/settings/tokens",
                )
            
            with gr.Row():
                test_hf_btn = gr.Button(
                    "🧪 Test Connection", variant="secondary"
                )

            hf_test_result = gr.Textbox(
                label="Test Result", interactive=False
            )

        # Training Parameters Section
        with gr.Group():
            gr.Markdown("### 🎯 Training Parameters")

            with gr.Row():
                batch_size = gr.Number(
                    label="Batch Size",
                    value=max(1, config_manager.get("training", "batch_size", 2) or 2),
                    minimum=1,
                    maximum=32,
                    step=1,
                    precision=0,
                )
                gradient_accum = gr.Number(
                    label="Gradient Accumulation Steps",
                    value=max(1, config_manager.get("training", "gradient_accumulation_steps", 4) or 4),
                    minimum=1,
                    maximum=32,
                    step=1,
                    precision=0,
                )

            with gr.Row():
                learning_rate = gr.Number(
                    label="Learning Rate",
                    value=max(1e-6, config_manager.get("training", "learning_rate", 2e-4) or 2e-4),
                    minimum=1e-6,
                    maximum=1e-2,
                    step=1e-5,
                )
                num_epochs = gr.Number(
                    label="Number of Epochs",
                    value=max(1, config_manager.get("training", "num_epochs", 1) or 1),
                    minimum=1,
                    maximum=10,
                    step=1,
                    precision=0,
                )

            with gr.Row():
                max_steps = gr.Number(
                    label="Max Steps",
                    value=max(1, config_manager.get("training", "max_steps", 60) or 60),
                    minimum=1,
                    maximum=10000,
                    step=1,
                    precision=0,
                )
                warmup_steps = gr.Number(
                    label="Warmup Steps",
                    value=max(0, config_manager.get("training", "warmup_steps", 5) or 5),
                    minimum=0,
                    maximum=100,
                    step=1,
                    precision=0,
                )
            
            with gr.Row():
                optimizer = gr.Dropdown(
                    label="Optimizer",
                    choices=["adamw_8bit", "adamw_torch", "sgd"],
                    value=config_manager.get("training", "optimizer", "adamw_8bit"),
                    scale=1
                )

                lr_scheduler = gr.Dropdown(
                    label="Learning Rate Scheduler",
                    choices=["linear", "cosine", "constant"],
                    value=config_manager.get("training", "lr_scheduler_type", "linear"),
                    scale=1
                )

        # LoRA Parameters Section
        with gr.Group():
            gr.Markdown("### 🔧 LoRA Configuration")

            with gr.Row():
                lora_r = gr.Number(
                    label="LoRA R (Rank)",
                    value=max(1, config_manager.get("lora", "r", 16) or 16),
                    minimum=1,
                    maximum=256,
                    step=1,
                    precision=0,
                    info="Higher = more parameters",
                    scale=1
                )
                lora_alpha = gr.Number(
                    label="LoRA Alpha",
                    value=max(1, config_manager.get("lora", "lora_alpha", 16) or 16),
                    minimum=1,
                    maximum=256,
                    step=1,
                    precision=0,
                    scale=1
                )
                lora_dropout = gr.Slider(
                    label="LoRA Dropout",
                    value=config_manager.get("lora", "lora_dropout", 0),
                    minimum=0,
                    maximum=0.5,
                    step=0.05,
                    scale=1
                )

        # Model Parameters Section
        with gr.Group():
            gr.Markdown("### 📊 Model Parameters")

            with gr.Row():
                max_seq_length = gr.Number(
                    label="Max Sequence Length",
                    value=max(128, config_manager.get("model", "max_seq_length", 2048) or 2048),
                    minimum=128,
                    maximum=8192,
                    step=128,
                    precision=0,
                    info="Maximum context window",
                    scale=1
                )

                load_in_4bit = gr.Checkbox(
                    label="Load in 4-bit",
                    value=config_manager.get("model", "load_in_4bit", True),
                    info="Use 4-bit quantization for lower VRAM usage",
                    scale=1
                )

        # Unsloth Configuration Section
        with gr.Group():
            gr.Markdown("### ⚡ Unsloth Configuration")

            with gr.Row():
                dataset_num_proc = gr.Number(
                    label="Dataset Num Processes",
                    value=max(1, config_manager.get("unsloth", "dataset_num_proc", 1) or 1),
                    minimum=1,
                    maximum=8,
                    step=1,
                    precision=0,
                    info="Number of processes for dataset preprocessing (Windows: must be 1)",
                    scale=1
                )

                packing = gr.Checkbox(
                    label="Enable Packing",
                    value=config_manager.get("unsloth", "packing", False),
                    info="Pack multiple sequences into one for efficiency",
                    scale=1
                )

            with gr.Row():
                use_gradient_checkpointing = gr.Checkbox(
                    label="Use Gradient Checkpointing",
                    value=config_manager.get("unsloth", "use_gradient_checkpointing", True),
                    info="Trade compute for memory (recommended for large models)",
                    scale=1
                )

                use_rslora = gr.Checkbox(
                    label="Use RS-LoRA",
                    value=config_manager.get("unsloth", "use_rslora", False),
                    info="Rank-Stabilized LoRA for better training stability",
                    scale=1
                )
            
                use_loftq = gr.Checkbox(
                    label="Use LoFTQ",
                    value=config_manager.get("unsloth", "use_loftq", False),
                    info="LoRA-Fine-Tuning-aware Quantization",
                    scale=1
                )

        # System Configuration Section
        with gr.Group():
            gr.Markdown("### 🖥️ System Configuration")

            with gr.Row():
                enable_debug_logs = gr.Checkbox(
                    label="Enable Debug Logs",
                    value=config_manager.get("system", "debug_logs", False),
                    info="Show detailed debug information in logs",
                    scale=1
                )

                num_gpus = gr.Number(
                    label="Number of GPUs",
                    value=max(1, config_manager.get("system", "num_gpus", 1) or 1),
                    minimum=1,
                    maximum=8,
                    step=1,
                    precision=0,
                    info="Number of GPUs to use for training (1 for single GPU)",
                    scale=1
                )

            with gr.Row():
                gpu_ids = gr.Textbox(
                    label="GPU IDs",
                    value=config_manager.get("system", "gpu_ids", "0"),
                    placeholder="0 or 0,1,2 for multi-GPU",
                    info="Comma-separated GPU device IDs (e.g., '0' or '0,1,2')",
                    scale=2
                )

        # Advanced Model Configuration Section
        with gr.Group():
            gr.Markdown("### 🧠 Advanced Model Configuration")

            with gr.Row():
                max_new_tokens = gr.Number(
                    label="Max New Tokens",
                    value=max(32, config_manager.get("model", "max_new_tokens", 256) or 256),
                    minimum=32,
                    maximum=2048,
                    step=32,
                    precision=0,
                    info="Maximum tokens to generate during inference",
                    scale=1
                )

                temperature = gr.Slider(
                    label="Temperature",
                    value=config_manager.get("model", "temperature", 0.7),
                    minimum=0.1,
                    maximum=2.0,
                    step=0.1,
                    info="Sampling temperature (lower = more focused, higher = more creative)",
                    scale=1
                )

            with gr.Row():
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    value=config_manager.get("model", "top_p", 0.9),
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    info="Cumulative probability for nucleus sampling",
                    scale=1
                )

                top_k = gr.Number(
                    label="Top K",
                    value=max(0, config_manager.get("model", "top_k", 50) or 50),
                    minimum=0,
                    maximum=200,
                    step=1,
                    precision=0,
                    info="Number of highest probability tokens to keep (0 = disabled)",
                    scale=1
                )

            with gr.Row():
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    value=config_manager.get("model", "repetition_penalty", 1.1),
                    minimum=1.0,
                    maximum=2.0,
                    step=0.05,
                    info="Penalty for repeating tokens (1.0 = no penalty)",
                    scale=1
                )

                use_cache = gr.Checkbox(
                    label="Use KV Cache",
                    value=config_manager.get("model", "use_cache", True),
                    info="Use key-value cache for faster generation",
                    scale=1
                )

        # Prompt Configuration Section
        with gr.Group():
            gr.Markdown("### 💬 AI Clone Prompt Configuration", elem_classes="section-header")
            gr.Markdown(
                "Configure how your AI clone should behave and respond. "
                "Use `{name}`, `{age}`, `{gender}`, `{location}`, `{country}`, `{hobbies}`, `{favorites}`, `{bio}` as placeholders."
            )

            system_prompt = gr.Textbox(
                label="System Prompt Template",
                value=config_manager.get("prompts", "system_prompt", ""),
                placeholder="Enter your custom system prompt template...",
                lines=15,
                info="This template defines the AI clone's personality and behavior"
            )

            response_structure = gr.Textbox(
                label="Response Structure Guidelines",
                value=config_manager.get("prompts", "response_structure", ""),
                placeholder="Enter response structure guidelines...",
                lines=3,
                info="Guidelines for how the AI clone should structure its responses"
            )

        # HuggingFace Upload Settings
        with gr.Group():
            gr.Markdown("### 📤 HuggingFace Upload Settings")

            with gr.Row():
                hf_repo = gr.Textbox(
                    label="Default Repository Name",
                    value=config_manager.get(
                        "huggingface", "default_repo", "my-charisma-model"
                    ),
                    placeholder="username/model-name",
                    scale=2
                )

                hf_private = gr.Checkbox(
                    label="Make Repository Private",
                    value=config_manager.get("huggingface", "private", True),
                    scale=1
                )

        # Save Button
        with gr.Row():
            save_btn = gr.Button("💾 Save All Settings", variant="primary", size="lg", scale=2)
            save_result = gr.Textbox(label="Save Result", interactive=False, scale=3)

        # Event Handlers
        def handle_save_config(
            notion_api_key,
            hf_api_token,
            batch_sz,
            grad_accum,
            lr,
            epochs,
            max_step,
            warmup,
            opt,
            lr_sched,
            r,
            alpha,
            dropout,
            max_seq,
            load_4bit,
            dataset_procs,
            pack,
            grad_checkpoint,
            rslora,
            loftq,
            debug_logs,
            n_gpus,
            gpus,
            max_tokens,
            temp,
            nucleus_p,
            topk,
            rep_penalty,
            kv_cache,
            sys_prompt,
            resp_structure,
            repo_name,
            private_repo,
        ):
            # Only save non-empty values for API credentials
            updates = {}
            
            # Notion API key (optional)
            if notion_api_key and notion_api_key.strip():
                updates["notion.api_key"] = notion_api_key.strip()
            
            # HuggingFace token (optional)
            if hf_api_token and hf_api_token.strip():
                updates["huggingface.token"] = hf_api_token.strip()
            
            # Training parameters (always save with defaults if empty)
            # Use max() to ensure minimum values are respected
            updates.update({
                "training.batch_size": max(1, int(batch_sz)) if batch_sz else 2,
                "training.gradient_accumulation_steps": max(1, int(grad_accum)) if grad_accum else 4,
                "training.learning_rate": float(lr) if lr else 2e-4,
                "training.num_epochs": max(1, int(epochs)) if epochs else 1,
                "training.max_steps": max(1, int(max_step)) if max_step else 60,
                "training.warmup_steps": max(0, int(warmup)) if warmup else 5,
                "training.optimizer": opt if opt else "adamw_8bit",
                "training.lr_scheduler_type": lr_sched if lr_sched else "linear",
                "lora.r": max(1, int(r)) if r else 16,
                "lora.lora_alpha": max(1, int(alpha)) if alpha else 16,
                "lora.lora_dropout": float(dropout) if dropout else 0.0,
                "model.max_seq_length": max(128, int(max_seq)) if max_seq else 2048,
                "model.load_in_4bit": bool(load_4bit),
                "unsloth.dataset_num_proc": max(1, int(dataset_procs)) if dataset_procs else 1,
                "unsloth.packing": bool(pack),
                "unsloth.use_gradient_checkpointing": bool(grad_checkpoint),
                "unsloth.use_rslora": bool(rslora),
                "unsloth.use_loftq": bool(loftq),
                "system.debug_logs": bool(debug_logs),
                "system.num_gpus": max(1, int(n_gpus)) if n_gpus else 1,
                "system.gpu_ids": gpus.strip() if gpus else "0",
                "model.max_new_tokens": max(32, int(max_tokens)) if max_tokens else 256,
                "model.temperature": float(temp) if temp else 0.7,
                "model.top_p": float(nucleus_p) if nucleus_p else 0.9,
                "model.top_k": max(0, int(topk)) if topk else 50,
                "model.repetition_penalty": float(rep_penalty) if rep_penalty else 1.1,
                "model.use_cache": bool(kv_cache),
                "prompts.system_prompt": sys_prompt if sys_prompt else "",
                "prompts.response_structure": resp_structure if resp_structure else "",
                "huggingface.default_repo": repo_name if repo_name else "my-charisma-model",
                "huggingface.private": bool(private_repo),
            })

            result = on_save_config(updates)

            if result["success"]:
                # Reload config to get fresh values
                config_manager.load_config()
                
                # Count how many credentials were saved
                cred_count = sum([
                    1 for k in updates.keys() 
                    if any(x in k for x in ["oauth_client_id", "oauth_client_secret", "api_key", "token"])
                ])
                cred_msg = f" ({cred_count} credential{'s' if cred_count != 1 else ''} saved)" if cred_count > 0 else ""
                message = f"✅ Settings saved successfully to charisma.toml{cred_msg}"
                
                # Return updated values to refresh all UI components
                return {
                    save_result: message,
                    batch_size: max(1, config_manager.get("training", "batch_size", 2) or 2),
                    gradient_accum: max(1, config_manager.get("training", "gradient_accumulation_steps", 4) or 4),
                    learning_rate: max(1e-6, config_manager.get("training", "learning_rate", 2e-4) or 2e-4),
                    num_epochs: max(1, config_manager.get("training", "num_epochs", 1) or 1),
                    max_steps: max(1, config_manager.get("training", "max_steps", 60) or 60),
                    warmup_steps: max(0, config_manager.get("training", "warmup_steps", 5) or 5),
                    optimizer: config_manager.get("training", "optimizer", "adamw_8bit"),
                    lr_scheduler: config_manager.get("training", "lr_scheduler_type", "linear"),
                    lora_r: max(1, config_manager.get("lora", "r", 16) or 16),
                    lora_alpha: max(1, config_manager.get("lora", "lora_alpha", 16) or 16),
                    lora_dropout: config_manager.get("lora", "lora_dropout", 0),
                    max_seq_length: max(128, config_manager.get("model", "max_seq_length", 2048) or 2048),
                    load_in_4bit: config_manager.get("model", "load_in_4bit", True),
                    dataset_num_proc: max(1, config_manager.get("unsloth", "dataset_num_proc", 1) or 1),
                    packing: config_manager.get("unsloth", "packing", False),
                    use_gradient_checkpointing: config_manager.get("unsloth", "use_gradient_checkpointing", True),
                    use_rslora: config_manager.get("unsloth", "use_rslora", False),
                    use_loftq: config_manager.get("unsloth", "use_loftq", False),
                    enable_debug_logs: config_manager.get("system", "debug_logs", False),
                    num_gpus: max(1, config_manager.get("system", "num_gpus", 1) or 1),
                    gpu_ids: config_manager.get("system", "gpu_ids", "0"),
                    max_new_tokens: max(32, config_manager.get("model", "max_new_tokens", 256) or 256),
                    temperature: config_manager.get("model", "temperature", 0.7),
                    top_p: config_manager.get("model", "top_p", 0.9),
                    top_k: max(0, config_manager.get("model", "top_k", 50) or 50),
                    repetition_penalty: config_manager.get("model", "repetition_penalty", 1.1),
                    use_cache: config_manager.get("model", "use_cache", True),
                    hf_repo: config_manager.get("huggingface", "default_repo", "my-charisma-model"),
                    hf_private: config_manager.get("huggingface", "private", True),
                }
            else:
                return {save_result: f"❌ Failed to save: {result['error']}"}

        save_btn.click(
            fn=handle_save_config,
            inputs=[
                notion_token,
                hf_token,
                batch_size,
                gradient_accum,
                learning_rate,
                num_epochs,
                max_steps,
                warmup_steps,
                optimizer,
                lr_scheduler,
                lora_r,
                lora_alpha,
                lora_dropout,
                max_seq_length,
                load_in_4bit,
                dataset_num_proc,
                packing,
                use_gradient_checkpointing,
                use_rslora,
                use_loftq,
                enable_debug_logs,
                num_gpus,
                gpu_ids,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                use_cache,
                system_prompt,
                response_structure,
                hf_repo,
                hf_private,
            ],
            outputs=[
                save_result,
                batch_size,
                gradient_accum,
                learning_rate,
                num_epochs,
                max_steps,
                warmup_steps,
                optimizer,
                lr_scheduler,
                lora_r,
                lora_alpha,
                lora_dropout,
                max_seq_length,
                load_in_4bit,
                dataset_num_proc,
                packing,
                use_gradient_checkpointing,
                use_rslora,
                use_loftq,
                enable_debug_logs,
                num_gpus,
                gpu_ids,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                use_cache,
                hf_repo,
                hf_private,
            ],
        )

        # Test Notion Connection
        def handle_test_notion(token):
            yield "🔄 Testing connection..."
            result = on_test_notion(token)
            if result["success"]:
                yield f"✅ Connected! Found {result['pages_count']} pages accessible"
            else:
                yield f"❌ Connection failed: {result['error']}"

        test_notion_btn.click(
            fn=handle_test_notion,
            inputs=notion_token,
            outputs=notion_test_result,
        )

        # Test HuggingFace Connection
        def handle_test_hf(token):
            yield "🔄 Testing connection..."
            result = on_test_hf(token)
            if result["success"]:
                yield f"✅ Connected as: {result['username']}"
            else:
                yield f"❌ Connection failed: {result['error']}"

        test_hf_btn.click(
            fn=handle_test_hf,
            inputs=hf_token,
            outputs=hf_test_result,
        )

    return settings_tab

"""HuggingFace upload tab for pushing fine-tuned models"""

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


def create_huggingface_tab(
    config_manager,
    on_push_to_hub: Callable,
) -> gr.Column:
    """Create the HuggingFace upload tab"""

    with gr.Column() as hf_tab:
        gr.Markdown("# 🤗 Push to Hugging Face Hub")
        gr.Markdown("Upload your fine-tuned model to Hugging Face for sharing and deployment")

        # Model Selection Section
        with gr.Group():
            gr.Markdown("### 📁 Model Selection", elem_classes="section-header")

            with gr.Row():
                local_model_path = gr.Textbox(
                    label="Local Model Path",
                    placeholder="./outputs/my-memory-clone",
                    value="./outputs/my-memory-clone",
                    info="Path to your fine-tuned model directory",
                )

        # Repository Settings Section
        with gr.Group():
            gr.Markdown("### 📦 Repository Settings", elem_classes="section-header")

            with gr.Row():
                repo_name = gr.Textbox(
                    label="Repository Name",
                    placeholder="username/my-charisma-clone",
                    value=config_manager.get("huggingface", "default_repo", "my-charisma-model"),
                    info="Format: username/model-name or organization/model-name",
                )
            
            with gr.Row():
                private_repo = gr.Checkbox(
                    label="Private Repository",
                    value=config_manager.get("huggingface", "private", True),
                    info="Keep your model private (only you can access it)",
                    scale=1
                )
                
                create_repo = gr.Checkbox(
                    label="Create Repository if it doesn't exist",
                    value=True,
                    info="Automatically create the repository if needed",
                    scale=1
                )

        # Model Card Section
        with gr.Group():
            gr.Markdown("### 📝 Model Card (README)", elem_classes="section-header")
            gr.Markdown("Provide information about your model for the README.md")

            model_description = gr.Textbox(
                label="Model Description",
                placeholder="This is my personal AI clone trained on my Notion memories...",
                lines=3,
                info="Describe what your model does"
            )

            with gr.Row():
                model_tags = gr.Textbox(
                    label="Tags (comma-separated)",
                    placeholder="charisma, personal-ai, memory-clone",
                    value="charisma, personal-ai, memory-clone, fine-tuned",
                    info="Tags to help others discover your model",
                )

        # Upload Options Section
        with gr.Group():
            gr.Markdown("### ⚙️ Upload Options", elem_classes="section-header")

            with gr.Row():
                commit_message = gr.Textbox(
                    label="Commit Message",
                    placeholder="Upload fine-tuned Charisma model",
                    value="Upload fine-tuned Charisma model",
                    info="Message describing this upload",
                )
            
            with gr.Row():
                include_training_args = gr.Checkbox(
                    label="Include Training Arguments",
                    value=True,
                    info="Upload training configuration with the model",
                    scale=1
                )
                
                include_tokenizer = gr.Checkbox(
                    label="Include Tokenizer",
                    value=True,
                    info="Upload tokenizer with the model (recommended)",
                    scale=1
                )

        # Push Button and Status
        push_btn = gr.Button(
            "Push to Hub", variant="primary", size="lg"
        )
        
        push_status = gr.Markdown(
            value="**Status:** Ready to upload",
            visible=True
        )

        # Progress Display
        push_progress = gr.Textbox(
            label="Upload Progress",
            lines=10,
            interactive=False,
            visible=False,
            placeholder="Upload progress will appear here..."
        )

        # Instructions
        with gr.Accordion("ℹ️ Instructions", open=False):
            gr.Markdown("""
            ### How to Push Your Model to Hugging Face Hub

            1. **Configure Token**: Make sure you've added your Hugging Face token in the Settings tab
            2. **Select Model**: Choose the local model path (e.g., `./outputs/my-memory-clone`)
            3. **Set Repository Name**: Format: `username/model-name` or `organization/model-name`
            4. **Privacy**: Choose whether the model should be private or public
            5. **Add Description**: Write a description and tags for your model
            6. **Push**: Click "Push to Hub" to upload

            ### What Gets Uploaded:
            - ✅ Model weights (LoRA adapters or full model)
            - ✅ Tokenizer files (if selected)
            - ✅ Training configuration (if selected)
            - ✅ Model card (README.md) with your description
            - ✅ Configuration files

            ### Requirements:
            - Valid Hugging Face token (get one at https://huggingface.co/settings/tokens)
            - Sufficient storage quota on Hugging Face (free tier: 10GB)
            - Completed training (model must exist in outputs folder)

            ### After Upload:
            You can find your model at: `https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_NAME`
            """)

        # Event Handler
        def handle_push(
            model_path,
            repo_id,
            is_private,
            create_if_missing,
            description,
            tags,
            commit_msg,
            include_train_args,
            include_tok
        ):
            """Push model to Hugging Face Hub"""
            try:
                # Show progress
                yield {
                    push_status: "**Status:** 🔄 Preparing to upload...",
                    push_progress: gr.update(visible=True, value="Initializing upload..."),
                }

                # Parse tags
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

                # Call push function
                for progress_update in on_push_to_hub(
                    model_path=model_path,
                    repo_id=repo_id,
                    private=is_private,
                    create_repo=create_if_missing,
                    description=description,
                    tags=tag_list,
                    commit_message=commit_msg,
                    include_training_args=include_train_args,
                    include_tokenizer=include_tok
                ):
                    if progress_update.get("success"):
                        repo_url = progress_update.get("repo_url", "")
                        final_status = f"**Status:** ✅ Successfully uploaded!\n\n**Repository:** [{repo_id}]({repo_url})"
                        yield {
                            push_status: final_status,
                            push_progress: gr.update(value=progress_update.get("message", "Upload complete!")),
                        }
                    elif progress_update.get("error"):
                        yield {
                            push_status: f"**Status:** ❌ Upload failed\n\n**Error:** {progress_update.get('error')}",
                            push_progress: gr.update(value=f"Error: {progress_update.get('error')}"),
                        }
                    else:
                        # Progress update
                        yield {
                            push_status: f"**Status:** 🔄 {progress_update.get('status', 'Uploading...')}",
                            push_progress: gr.update(value=progress_update.get("message", "")),
                        }

            except Exception as e:
                logger.error(f"Push to hub error: {e}")
                yield {
                    push_status: f"**Status:** ❌ Error: {str(e)}",
                    push_progress: gr.update(value=f"Error: {str(e)}"),
                }

        push_btn.click(
            fn=handle_push,
            inputs=[
                local_model_path,
                repo_name,
                private_repo,
                create_repo,
                model_description,
                model_tags,
                commit_message,
                include_training_args,
                include_tokenizer,
            ],
            outputs=[push_status, push_progress]
        )

    return hf_tab

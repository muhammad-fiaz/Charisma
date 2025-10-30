"""Main training tab for Gradio UI"""

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


def create_main_tab(
    on_connect_notion: Callable,
    on_select_memories: Callable,
    on_generate: Callable,
    on_stop_training: Callable,
    on_get_cached_memories: Callable,  # New callback
    available_models: list,
    default_model: str,
    training_active: bool = False,  # New parameter
) -> gr.Column:
    """Create the main training tab"""

    with gr.Column() as main_tab:
        gr.Markdown("# Charisma - Personal Memory Clone", elem_classes=["centered-title"])
        gr.Markdown("Create an AI clone of yourself using your memories from Notion", elem_classes=["centered-subtitle"])

        # Personal Information Section
        with gr.Group():
            gr.Markdown("### Personal Information")
            
            with gr.Row():
                name_input = gr.Textbox(
                    label="Name", 
                    placeholder="John Doe",
                    value="John Doe",
                    scale=2
                )
                age_input = gr.Number(
                    label="Age", 
                    minimum=1, 
                    maximum=150, 
                    scale=1, 
                    precision=0, 
                    value=25
                )

            with gr.Row():
                country_input = gr.Textbox(
                    label="Country", 
                    placeholder="United States",
                    value="United States",
                    scale=1
                )
                location_input = gr.Textbox(
                    label="Location", 
                    placeholder="New York",
                    value="New York",
                    scale=1
                )

            with gr.Row():
                hobbies_input = gr.Textbox(
                    label="Hobbies",
                    placeholder="Reading, Coding, Photography",
                    value="Reading, Coding, Photography",
                    scale=1,
                )
                favorites_input = gr.Textbox(
                    label="Favorites",
                    placeholder="Pizza, Sci-fi movies, Python",
                    value="Pizza, Sci-fi movies, Python",
                    scale=1,
                )

        # Notion Connection Section
        with gr.Group():
            gr.Markdown("### 📝 Notion Memories")
            
            notion_status = gr.Markdown(
                value="**Status:** Not connected"
            )
            
            with gr.Row():
                connect_btn = gr.Button(
                    "🔗 Connect to Notion", variant="secondary", scale=1
                )
            
            with gr.Row():
                refresh_btn = gr.Button(
                    "� Refresh Memories", variant="secondary", scale=1, visible=False
                )
            
            with gr.Row():
                memories_count = gr.Number(
                    label="Total Memories Found", value=0, interactive=False, visible=False
                )
        
        no_memories_msg = gr.Markdown(
            "ℹ️ No memories found. Make sure your Notion pages are accessible to the integration.",
            visible=False
        )

        memories_selector = gr.CheckboxGroup(
            label="Select Memories to Use",
            choices=[],
            visible=False,
            info="Select the memories you want to use for training",
            elem_classes=["scrollable-checkboxgroup"]
        )        # Model Configuration Section
        with gr.Group():
            gr.Markdown("### 🤖 Model Configuration")

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Select Model",
                    choices=[m["id"] for m in available_models],
                    value=default_model,
                    info="Choose a pre-configured model or enter a custom HuggingFace model ID",
                    scale=2
                )

                custom_model_input = gr.Textbox(
                    label="Custom Model (optional)",
                    placeholder="e.g., username/custom-model",
                    info="Leave empty to use selected model above",
                    scale=2
                )

            model_info = gr.Markdown(
                "**Model:** unsloth/gemma-3-270m-it | **VRAM:** ~4GB | **Params:** 270M"
            )

            training_mode = gr.Radio(
                label="Training Mode",
                choices=[
                    "LoRA Fine-tune (Fast, Low VRAM)",
                    "Full Fine-tune (Slow, High VRAM)",
                ],
                value="LoRA Fine-tune (Fast, Low VRAM)",
                info="LoRA is recommended for most users",
            )

        # Generation Section
        with gr.Group():
            gr.Markdown("### 🚀 Generate Your AI Clone")

            with gr.Row():
                output_name = gr.Textbox(
                    label="Output Model Name",
                    placeholder="my-memory-clone",
                    value="my-memory-clone",
                    info="Name for your fine-tuned model",
                )
            
            # Console output display
            gr.Markdown("#### 📊 Training Console Output")
            console_output = gr.Code(
                label="Live Training Logs",
                language="shell",
                lines=15,
                value="Waiting for training to start...\nConsole logs will appear here in real-time.",
                interactive=False,
                elem_id="training-console",
            )
            
            generate_btn = gr.Button(
                "Generate AI Clone", variant="primary", size="lg"
            )
            
            stop_generate_btn = gr.Button(
                "Stop Training", variant="stop", size="lg", interactive=training_active
            )

            result_output = gr.Markdown(visible=False)

        # Event Handlers
        def update_model_info(selected_model, custom_model):
            if custom_model.strip():
                return f"**Model:** {custom_model} | **VRAM:** Unknown | **Params:** Unknown"

            model_id = selected_model
            for model in available_models:
                if model["id"] == model_id:
                    return f"**Model:** {model['name']} | **VRAM:** {model['vram']} | **Params:** {model['params']}"

            return (
                f"**Model:** {selected_model} | **VRAM:** Unknown | **Params:** Unknown"
            )

        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown, custom_model_input],
            outputs=model_info,
        )

        custom_model_input.change(
            fn=update_model_info,
            inputs=[model_dropdown, custom_model_input],
            outputs=model_info,
        )

        # Connect to Notion
        def handle_notion_connect():
            # Show loading state
            yield (
                "🔄 Connecting to Notion...",
                gr.update(interactive=False),  # Disable connect button
                gr.update(visible=False),  # Hide refresh button
                gr.update(visible=False),  # Hide count
                gr.update(visible=False),  # Hide no memories msg
                gr.update(visible=False),  # Hide selector
            )
            
            result = on_connect_notion()
            
            if result["success"]:
                # Create detailed status message
                summary = result.get("summary", "")
                total_items = result.get("total_items", len(result.get("memories", [])))
                memory_count = len(result.get("memories", []))
                
                status_msg = f"✅ Connected to Notion\n\n{summary}" if summary else f"✅ Connected to Notion - Found {total_items} items"
                
                # Show different UI based on whether memories were found
                if memory_count > 0:
                    yield (
                        status_msg,
                        gr.update(interactive=True),  # Re-enable connect button
                        gr.update(visible=True),
                        gr.update(visible=True, value=memory_count),
                        gr.update(visible=False),  # Hide no memories message
                        gr.update(
                            visible=True,
                            choices=result["memory_choices"],
                            value=result["memory_choices"],
                        ),
                    )
                else:
                    yield (
                        status_msg,
                        gr.update(interactive=True),  # Re-enable connect button
                        gr.update(visible=True),
                        gr.update(visible=True, value=0),
                        gr.update(visible=True),  # Show no memories message
                        gr.update(visible=False, choices=[]),  # Hide empty selector
                    )
            else:
                yield (
                    f"❌ Connection failed: {result['error']}",
                    gr.update(interactive=True),  # Re-enable connect button
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

        connect_btn.click(
            fn=handle_notion_connect,
            outputs=[notion_status, connect_btn, refresh_btn, memories_count, no_memories_msg, memories_selector],
        )

        # Refresh memories
        refresh_btn.click(
            fn=handle_notion_connect,
            outputs=[notion_status, connect_btn, refresh_btn, memories_count, no_memories_msg, memories_selector],
        )

        # Generate AI Clone
        def handle_generate(
            name,
            age,
            country,
            location,
            hobbies,
            favorites,
            selected_memories,
            selected_model,
            custom_model,
            training_mode_choice,
            output_model_name,
        ):
            # Show loading state
            yield (
                gr.update(interactive=False, value="Generating..."),  # Disable button
                gr.update(visible=False),  # Hide result
                "Starting AI clone generation...\nThis may take several minutes...\n",  # Update console
                gr.update(interactive=True),  # Enable stop button
            )
            
            personal_info = {
                "name": name,
                "age": str(int(age)) if age else "",
                "country": country,
                "location": location,
                "hobbies": hobbies,
                "favorites": favorites,
            }

            model_to_use = (
                custom_model.strip() if custom_model.strip() else selected_model
            )
            use_full_finetune = "Full Fine-tune" in training_mode_choice

            result = on_generate(
                personal_info=personal_info,
                selected_memories=selected_memories,
                model_name=model_to_use,
                use_full_finetune=use_full_finetune,
                output_name=output_model_name,
            )

            if result["success"]:
                result_md = f"""
## ✅ Success!

Your AI memory clone has been created successfully!

**Model saved to:** `{result["output_dir"]}`

**Training Statistics:**
- Total memories used: {result["stats"]["total_memories"]}
- Training steps: {result["stats"]["steps"]}
- Final loss: {result["stats"]["final_loss"]:.4f}

**Next Steps:**
1. Test your model in the inference section
2. Push to HuggingFace Hub from Settings tab
3. Use your AI clone for conversations!
"""
                yield (
                    gr.update(interactive=True, value="Generate AI Clone"),  # Re-enable button
                    gr.update(visible=True, value=result_md),
                    "Training completed successfully!\n\nCheck the results above.",
                    gr.update(interactive=False),  # Disable stop button
                )
            else:
                error_md = f"""
## Error

Training failed: {result["error"]}

Please check the Logs tab for more details.
"""
                yield (
                    gr.update(interactive=True, value="Generate AI Clone"),  # Re-enable button
                    gr.update(visible=True, value=error_md),
                    f"Error: {result['error']}\n\nPlease check the Logs tab for details.",
                    gr.update(interactive=False),  # Disable stop button
                )

        generate_btn.click(
            fn=handle_generate,
            inputs=[
                name_input,
                age_input,
                country_input,
                location_input,
                hobbies_input,
                favorites_input,
                memories_selector,
                model_dropdown,
                custom_model_input,
                training_mode,
                output_name,
            ],
            outputs=[generate_btn, result_output, console_output, stop_generate_btn],
        )

        stop_generate_btn.click(
            fn=on_stop_training,
            inputs=None,
            outputs=None
        )

    return main_tab

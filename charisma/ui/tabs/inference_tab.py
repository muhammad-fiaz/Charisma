"""Inference tab for testing fine-tuned models"""

from typing import Callable, Optional
import os

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.utils.logger import get_logger

logger = get_logger()


def create_inference_tab(
    on_load_model: Callable,
    on_inference: Callable,
    on_stop_inference: Callable,
    inference_active: bool = False,  # New parameter
) -> gr.Column:
    """Create the inference tab for testing fine-tuned models"""

    with gr.Column() as inference_tab:
        gr.Markdown("# 🤖 Test Your AI Clone")
        gr.Markdown("Load and test your fine-tuned AI clone models")

        # Model Selection Section
        with gr.Group():
            gr.Markdown("### 📁 Model Selection", elem_classes="section-header")

            with gr.Row():
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="./outputs/my-memory-clone",
                    value="./outputs/my-memory-clone",
                    info="Path to your fine-tuned model directory",
                    scale=3
                )
            
            with gr.Row():
                load_model_btn = gr.Button(
                    "📥 Load Model", variant="primary", size="lg"
                )
            
            load_status = gr.Markdown(
                value="**Status:** No model loaded", 
                visible=True
            )

        # Inference Parameters Section
        with gr.Group():
            gr.Markdown("### ⚙️ Inference Parameters", elem_classes="section-header")

            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    info="Higher = more creative, Lower = more focused",
                    scale=1
                )
                
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    info="Probability threshold for token selection",
                    scale=1
                )
            
            with gr.Row():
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=32,
                    maximum=2048,
                    value=256,
                    step=32,
                    info="Maximum length of generated response",
                    scale=1
                )
                
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05,
                    info="Penalty for repeating tokens",
                    scale=1
                )

        # Chat Interface Section
        with gr.Group():
            gr.Markdown("### Chat with Your AI Clone", elem_classes="section-header")

            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=True,
                avatar_images=(None, None),
                type="messages"
            )

            with gr.Row():
                user_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    scale=4
                )
            
            submit_btn = gr.Button("Send", variant="primary", size="lg")
            stop_btn = gr.Button("Stop", variant="stop", size="lg", interactive=inference_active)
            clear_btn = gr.Button("Clear Chat", size="lg")

        # System Prompt Override (Optional)
        with gr.Accordion("Advanced: Override System Prompt", open=False):
            system_prompt_override = gr.Textbox(
                label="Custom System Prompt",
                placeholder="Leave empty to use the default system prompt from training...",
                lines=5,
                info="Override the system prompt for this conversation (optional)"
            )

        # Event Handlers
        def handle_load_model(model_path_input):
            """Load the fine-tuned model"""
            try:
                result = on_load_model(model_path_input)
                if result.get("success"):
                    status = f"**Status:** ✅ Model loaded successfully from `{model_path_input}`\n\n"
                    status += f"**Model Type:** {result.get('model_type', 'Unknown')}\n"
                    status += f"**Ready for inference!**"
                    return {
                        load_status: status,
                    }
                else:
                    return {
                        load_status: f"**Status:** ❌ Failed to load model\n\n**Error:** {result.get('error', 'Unknown error')}"
                    }
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return {
                    load_status: f"**Status:** ❌ Error: {str(e)}"
                }

        def handle_inference(
            message, 
            history, 
            temp, 
            top_p_val, 
            max_tokens, 
            rep_penalty,
            sys_prompt_override
        ):
            """Generate response from the AI clone"""
            if not message.strip():
                return history, ""
            
            # Add user message to history (using messages format)
            history = history or []
            history.append({"role": "user", "content": message})
            
            try:
                # Call inference function
                result = on_inference(
                    message=message,
                    temperature=temp,
                    top_p=top_p_val,
                    max_new_tokens=int(max_tokens),
                    repetition_penalty=rep_penalty,
                    system_prompt=sys_prompt_override if sys_prompt_override else None
                )
                
                if result.get("success"):
                    response = result.get("response", "")
                    history.append({"role": "assistant", "content": response})
                else:
                    error_msg = result.get("error", "Unknown error")
                    history.append({"role": "assistant", "content": f"Error: {error_msg}"})
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            
            return history, ""

        def handle_clear():
            """Clear the chat history"""
            return [], ""

        # Wire up events
        load_model_btn.click(
            fn=handle_load_model,
            inputs=[model_path],
            outputs=[load_status]
        )

        submit_btn.click(
            fn=handle_inference,
            inputs=[
                user_input, 
                chatbot, 
                temperature, 
                top_p, 
                max_new_tokens, 
                repetition_penalty,
                system_prompt_override
            ],
            outputs=[chatbot, user_input]
        )

        user_input.submit(
            fn=handle_inference,
            inputs=[
                user_input, 
                chatbot, 
                temperature, 
                top_p, 
                max_new_tokens, 
                repetition_penalty,
                system_prompt_override
            ],
            outputs=[chatbot, user_input]
        )

        stop_btn.click(
            fn=on_stop_inference,
            inputs=None,
            outputs=None
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=None,
            outputs=[chatbot, user_input]
        )

    return inference_tab

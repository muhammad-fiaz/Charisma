"""Inference tab for testing fine-tuned models"""

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


def create_inference_tab(
    on_load_model: Callable,
    on_inference: Callable,
    on_stop_inference: Callable,
    inference_active: bool = False,
    get_personal_info: Callable = None,
) -> gr.Column:
    """Create the inference tab for testing fine-tuned models"""

    with gr.Column() as inference_tab:
        gr.Markdown("# 🤖 Test Your AI Clone")
        gr.Markdown(
            """
            Your AI clone has been trained to embody your personality, memories, and way of speaking.
            
            **What to expect:**
            - ✅ The model responds as YOU - in first person (I, me, my)
            - ✅ References YOUR memories naturally
            - ✅ Uses YOUR personality and speech patterns
            - ✅ No system prompts needed - personality is embedded in the model
            
            **Try saying:** "Hi!", "What's up?", "Tell me about yourself", "What did you do recently?"
            """
        )

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
                    value=1.0,  # Gemma-3 recommended default
                    step=0.1,
                    info="Higher = more creative, Lower = more focused (Gemma-3 default: 1.0)",
                    scale=1
                )
                
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,  # Gemma-3 recommended default
                    step=0.05,
                    info="Probability threshold for token selection (Gemma-3 default: 0.95)",
                    scale=1
                )
            
            with gr.Row():
                top_k = gr.Slider(
                    label="Top K",
                    minimum=0,
                    maximum=200,
                    value=64,  # Gemma-3 recommended default
                    step=1,
                    info="Number of top tokens to consider (Gemma-3 default: 64, 0=disabled)",
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

        # System Prompt Override (Optional - Advanced Testing Only)
        with gr.Accordion("⚠️ Advanced: System Prompt Override (Optional)", open=False):
            gr.Markdown(
                """
                **Note:** Your AI clone's personality is already embedded in the model from training.
                
                Leave this empty for normal testing. Only use this field if you want to experiment with different system prompts for testing purposes.
                """
            )
            system_prompt_override = gr.Textbox(
                label="Custom System Prompt (Leave Empty for Raw Model Inference)",
                placeholder="",
                lines=5,
                info="⚠️ Advanced users only - Leave empty to test the model's natural personality"
            )

        # Event Handlers
        def handle_load_model(model_path_input):
            """Load the fine-tuned model"""
            try:
                result = on_load_model(model_path_input)
                if result.get("success"):
                    status = f"**Status:** ✅ Model loaded successfully from `{model_path_input}`\n\n"
                    status += f"**Model Type:** {result.get('model_type', 'Unknown')}\n"
                    status += "**Ready for inference!**"
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
            top_k_val,
            rep_penalty,
            max_tokens,
            sys_prompt_override
        ):
            """Generate response from the AI clone"""
            if not message.strip():
                return history, ""
            
            # Add user message to history (using messages format)
            history = history or []
            history.append({"role": "user", "content": message})
            
            try:
                # Get personal info for system prompt
                personal_info = get_personal_info() if get_personal_info else None
                
                # Call inference function
                result = on_inference(
                    message=message,
                    temperature=temp,
                    top_p=top_p_val,
                    top_k=int(top_k_val),
                    max_new_tokens=int(max_tokens),
                    repetition_penalty=rep_penalty,
                    system_prompt=sys_prompt_override if sys_prompt_override else None,
                    personal_info=personal_info
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
                top_k,
                repetition_penalty,
                max_new_tokens,
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
                top_k,
                repetition_penalty,
                max_new_tokens,
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

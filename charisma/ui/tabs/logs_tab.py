"""Logs tab for Gradio UI"""

from pathlib import Path

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.utils.logger import get_logger

logger = get_logger()


def create_logs_tab(log_dir: str = "./logs") -> gr.Column:
    """Create the logs tab"""

    with gr.Column() as logs_tab:
        gr.Markdown("# 📋 Logs")
        gr.Markdown("View application logs and training progress")

        with gr.Row():
            log_files_dropdown = gr.Dropdown(
                label="Select Log File",
                choices=[],
                interactive=True,
                scale=3,
            )
            refresh_logs_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)

        log_content = gr.Textbox(
            label="Log Content",
            lines=25,
            interactive=False,
            max_lines=50,
            show_copy_button=True,
        )

        # Event Handlers
        def get_log_files():
            """Get list of log files"""
            log_path = Path(log_dir)
            if not log_path.exists():
                return []

            log_files = sorted(
                log_path.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True
            )
            return [f.name for f in log_files]

        def load_log_file(filename):
            """Load content of selected log file"""
            if not filename:
                return "No log file selected"

            log_path = Path(log_dir) / filename

            if not log_path.exists():
                return f"Log file not found: {filename}"

            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # If file is too large, show only last 10000 lines
                lines = content.split("\n")
                if len(lines) > 10000:
                    content = "\n".join(lines[-10000:])
                    content = f"[Showing last 10000 lines]\n\n{content}"

                return content
            except Exception as e:
                return f"Error reading log file: {str(e)}"

        def refresh_logs():
            """Refresh log files list"""
            files = get_log_files()
            if files:
                return gr.update(choices=files, value=files[0]), load_log_file(files[0])
            else:
                return gr.update(choices=[], value=None), "No log files found"

        # Initialize with current log files
        refresh_logs_btn.click(
            fn=refresh_logs,
            outputs=[log_files_dropdown, log_content],
        )

        log_files_dropdown.change(
            fn=load_log_file,
            inputs=log_files_dropdown,
            outputs=log_content,
        )

    return logs_tab

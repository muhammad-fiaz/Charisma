"""Logger utility for Charisma project using Logly with Rich integration"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from logly import logger as _logly_logger
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

try:
    from rich.console import Console
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not available. Falling back to basic colors.", file=sys.stderr)

# Flag to track if logger has been configured
_configured = False


def _create_rich_callback():
    """Create Rich-based color callback for enhanced terminal styling"""
    if not RICH_AVAILABLE:
        return None
    
    def rich_color_callback(level: str, text: str) -> str:
        """Rich-based color callback with advanced styling and emojis"""
        # Create Rich Text object
        rich_text = Text(text)
        
        # Apply styling based on level with emojis
        level_styles = {
            "TRACE": ("dim cyan", "🔍"),
            "DEBUG": ("blue", "🐛"),
            "INFO": ("white", "ℹ️"),
            "SUCCESS": ("bold green", "✅"),
            "WARNING": ("bold yellow", "⚠️"),
            "ERROR": ("bold red", "❌"),
            "CRITICAL": ("bold white on red", "🚨"),
            "FAIL": ("bold magenta", "✗"),
        }
        
        style_config = level_styles.get(level, ("white", ""))
        style, emoji = style_config
        
        # Add emoji prefix if available
        if emoji:
            rich_text = Text(f"{emoji} {text}")
        
        rich_text.stylize(style)
        
        # Convert to ANSI and return
        try:
            from io import StringIO
            buffer = StringIO()
            temp_console = Console(file=buffer, force_terminal=True)
            temp_console.print(rich_text, end="")
            return buffer.getvalue()
        except Exception:
            # Fallback to plain text on error
            return text
    
    return rich_color_callback


def setup_logger(log_file: Optional[str] = None, debug: bool = False) -> None:
    """Setup Logly logger with Rich integration and file output
    
    Args:
        log_file: Optional custom log file path
        debug: Enable debug-level logging (default: False, INFO only)
    """
    global _configured
    
    if _configured:
        return
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        log_file = str(
            log_dir / f"charisma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    
    # Create Rich color callback if available
    color_callback = _create_rich_callback()
    
    # Set log level based on debug flag
    log_level = "DEBUG" if debug else "INFO"
    
    # Configure Logly with Rich integration
    config = {
        "level": log_level,
        "color": True,
        "auto_sink": True,  # Console output with Rich colors
        "auto_sink_levels": {
            "DEBUG": {
                "path": log_file,
                "rotation": "daily",
                "retention": 7,
                "date_enabled": True,
                "async_write": True,
            },
            "INFO": {
                "path": str(log_dir / "app.log"),
                "rotation": "daily",
                "retention": 30,
                "async_write": True,
            },
            "ERROR": {
                "path": str(log_dir / "errors.log"),
                "rotation": "daily",
                "retention": 90,
                "json": True,
                "async_write": True,
            },
        },
    }
    
    # Add Rich callback if available
    if color_callback:
        config["color_callback"] = color_callback
    else:
        # Fallback to default Logly colors
        config["level_colors"] = {
            "TRACE": "CYAN",
            "DEBUG": "BLUE",
            "INFO": "WHITE",
            "SUCCESS": "BRIGHT_GREEN",
            "WARNING": "BRIGHT_YELLOW",
            "ERROR": "BRIGHT_RED",
            "CRITICAL": "BRIGHT_MAGENTA",
            "FAIL": "MAGENTA",
        }
    
    _logly_logger.configure(**config)
    
    _configured = True


def get_logger():
    """Get the global Logly logger instance"""
    if not _configured:
        setup_logger()
    return _logly_logger

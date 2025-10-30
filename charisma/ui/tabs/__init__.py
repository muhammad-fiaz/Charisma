"""UI tabs package exports"""

from charisma.ui.tabs.main_tab import create_main_tab
from charisma.ui.tabs.settings_tab import create_settings_tab
from charisma.ui.tabs.logs_tab import create_logs_tab
from charisma.ui.tabs.inference_tab import create_inference_tab
from charisma.ui.tabs.huggingface_tab import create_huggingface_tab

__all__ = [
    "create_main_tab", 
    "create_settings_tab", 
    "create_logs_tab",
    "create_inference_tab",
    "create_huggingface_tab"
]

#!/usr/bin/env python3
"""
Charisma Launcher
Simple launcher script for the Charisma application
"""

import sys

try:
    from charisma import main
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )



if __name__ == "__main__":
    sys.exit(main())

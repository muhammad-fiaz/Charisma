"""Version information for Charisma"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("charisma")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback for development

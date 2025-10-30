"""HuggingFace Hub client for model management"""

from typing import Optional

try:
    from huggingface_hub import HfApi, login, whoami
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma.utils.logger import get_logger

logger = get_logger()


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub"""

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.api = None
        self.logged_in = False

        if api_token:
            self.login(api_token)

    def login(self, token: str) -> bool:
        """Login to HuggingFace Hub"""
        try:
            login(token=token)
            self.api = HfApi()
            self.api_token = token
            self.logged_in = True
            logger.info("Successfully logged in to HuggingFace Hub")
            return True
        except Exception as e:
            logger.error("Failed to login to HuggingFace Hub: %s", str(e))
            self.logged_in = False
            return False

    def test_connection(self) -> bool:
        """Test HuggingFace Hub connection"""
        if not self.logged_in:
            return False
        try:
            whoami(token=self.api_token)
            logger.info("HuggingFace Hub connection test successful")
            return True
        except Exception as e:
            logger.error("HuggingFace Hub connection test failed: %s", str(e))
            return False

    def push_model(self, repo_id: str, folder_path: str, private: bool = False) -> bool:
        """Push model to HuggingFace Hub"""
        if not self.logged_in or not self.api:
            logger.error("Not logged in to HuggingFace Hub")
            return False

        try:
            self.api.create_repo(
                repo_id=repo_id, private=private, exist_ok=True, repo_type="model"
            )
            self.api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload model from Charisma",
            )
            logger.info("Successfully pushed model to %s", repo_id)
            return True
        except Exception as e:
            logger.error("Failed to push model: %s", str(e))
            return False

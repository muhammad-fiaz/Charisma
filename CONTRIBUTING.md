# Contributing to Charisma

Thank you for your interest in contributing to **Charisma**! 🎉

We welcome contributions from everyone, whether it's bug reports, feature requests, documentation improvements, or code contributions.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to be respectful and constructive. We want to maintain a welcoming and inclusive environment for everyone.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```powershell
   git clone https://github.com/YOUR_USERNAME/charisma.git
   cd charisma
   ```
3. **Set up the development environment** (see below)

## Development Setup

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA (for testing training)
- Git

### Installation

```powershell
# Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Project Structure

```
charisma/
├── charisma/              # Main package
│   ├── config/           # Configuration management
│   ├── core/             # Core training logic
│   ├── integrations/     # External API integrations
│   ├── ui/               # Gradio UI components
│   └── utils/            # Utility functions
├── tests/                # Test files (to be added)
├── docs/                 # Documentation (to be added)
└── examples/             # Example scripts (to be added)
```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- **Clear title** describing the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs. **actual behavior**
- **Environment details** (OS, Python version, GPU, etc.)
- **Error messages** and logs if applicable

### Suggesting Features

Feature requests are welcome! Please create an issue with:

- **Clear description** of the feature
- **Use case** - why this feature would be useful
- **Possible implementation** (if you have ideas)

### Code Contributions

We welcome code contributions! Here's how:

1. **Check existing issues** to see if someone is already working on it
2. **Create an issue** if one doesn't exist
3. **Comment on the issue** to let others know you're working on it
4. **Fork and create a branch** for your work
5. **Make your changes** following our coding standards
6. **Test your changes** thoroughly
7. **Submit a pull request**

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```powershell
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, descriptive commits:
   ```powershell
   git commit -m "Add feature: description of what you did"
   ```

3. **Push to your fork**:
   ```powershell
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what was changed and why
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots if UI changes are involved

5. **Respond to feedback** during code review

6. **Wait for approval** from maintainers

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** where appropriate
- Use **docstrings** for all functions, classes, and modules
- Keep lines under **120 characters**

Example:

```python
def process_data(input_data: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """
    Process input data and filter by threshold.
    
    Args:
        input_data: List of dictionaries containing data
        threshold: Minimum value to filter (default: 0.5)
    
    Returns:
        Filtered list of dictionaries
    """
    return [item for item in input_data if item.get("value", 0) > threshold]
```

### Code Organization

- **One class per file** (unless closely related)
- **Group imports** (standard library, third-party, local)
- **Use meaningful variable names**
- **Add comments** for complex logic
- **Keep functions small** and focused

### Logging

Use the project's logger:

```python
from charisma.utils.logger import get_logger

logger = get_logger()

logger.info("Processing started")
logger.warning("Potential issue detected")
logger.error("An error occurred: %s", error_message)
```

### Error Handling

- Use **specific exceptions** (not bare `except`)
- **Log errors** with context
- **Provide helpful error messages** to users

Example:

```python
try:
    result = process_data(data)
except ValueError as e:
    logger.error("Invalid data format: %s", str(e))
    raise ValueError(f"Data processing failed: {str(e)}")
```

## Testing

(To be implemented)

We plan to add comprehensive tests. For now:

1. **Manual testing** - Test your changes thoroughly
2. **Check different scenarios** - Edge cases, error conditions
3. **Test on GPU and CPU** if applicable
4. **Verify UI changes** in the Gradio interface

Future testing framework:
- `pytest` for unit tests
- `pytest-cov` for coverage
- Integration tests for workflows

## Documentation

### Code Documentation

- **Docstrings** for all public functions and classes
- **Comments** for complex logic
- **Type hints** for function signatures

### User Documentation

If you add a new feature, please update:

- **README.md** - If it affects setup or usage
- **QUICKSTART.md** - If it affects the quick start process
- **Configuration comments** in `charisma.toml`

### Example Documentation

```python
class DataProcessor:
    """
    Processes user data and memories into training format.
    
    This class handles the conversion of raw Notion memories and personal
    information into the format required for model training.
    
    Attributes:
        None
    
    Example:
        >>> processor = DataProcessor()
        >>> dataset = processor.create_training_dataset(personal_info, memories)
    """
    
    def create_training_dataset(self, personal_info: Dict[str, str], 
                                memories: List[Dict]) -> Dataset:
        """
        Create training dataset from personal info and memories.
        
        Args:
            personal_info: Dictionary with keys 'name', 'age', 'location', etc.
            memories: List of memory dictionaries with 'title', 'content', 'date'
        
        Returns:
            HuggingFace Dataset object ready for training
        
        Raises:
            ValueError: If personal_info or memories are empty
        """
        pass
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

- **feat:** New feature (e.g., `feat: add support for custom models`)
- **fix:** Bug fix (e.g., `fix: resolve CUDA memory leak`)
- **docs:** Documentation (e.g., `docs: update README with new examples`)
- **style:** Code style changes (e.g., `style: format code with black`)
- **refactor:** Code refactoring (e.g., `refactor: simplify data processor`)
- **test:** Adding tests (e.g., `test: add unit tests for trainer`)
- **chore:** Maintenance (e.g., `chore: update dependencies`)

## Areas for Contribution

Here are some areas where we'd love contributions:

### High Priority
- [ ] Unit tests and integration tests
- [ ] Better error handling and validation
- [ ] Performance optimizations
- [ ] Memory usage improvements

### Features
- [ ] Support for more data sources (Google Docs, Markdown files, etc.)
- [ ] Advanced inference UI with chat interface
- [ ] Model comparison tools
- [ ] Automatic hyperparameter tuning
- [ ] Multi-GPU training support

### Documentation
- [ ] Video tutorials
- [ ] More examples and use cases
- [ ] API documentation
- [ ] Troubleshooting guide

### UI/UX
- [ ] Dark mode theme
- [ ] Progress bars and better visual feedback
- [ ] Model performance metrics display
- [ ] Export/import settings functionality

## Questions?

If you have questions about contributing:

- **Open a discussion** on GitHub
- **Email:** contact@muhammadfiaz.com
- **Check existing issues** for similar questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Charisma! 🧠✨

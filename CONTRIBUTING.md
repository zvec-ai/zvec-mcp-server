# Contributing to Zvec MCP Server

Thank you for your interest in contributing to Zvec MCP Server! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10 - 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup Steps

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/yourusername/zvec-mcp-server.git
cd zvec-mcp-server
```

2. Create a virtual environment and install dependencies:
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. Verify the setup by running tests:
```bash
pytest tests/ -v
```

## Development Workflow

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check code style
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_server.py -v

# Run with coverage
pytest tests/ --cov=zvec_mcp --cov-report=html

# Run specific test class
pytest tests/test_server.py::TestMultiVectorQuery -v
```

### Testing with MCP Inspector

```bash
# Start the MCP Inspector
npx @modelcontextprotocol/inspector python -m zvec_mcp
```

## Submitting Changes

### Pull Request Process

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   ```

5. **Update documentation** if needed (README.md, docstrings, etc.)

6. **Commit your changes** with a clear commit message:
   ```bash
   git commit -m "feat: add new embedding search tool"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request** against the main repository

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, no logic change)
- `refactor:` - Code refactoring
- `test:` - Test changes
- `chore:` - Build process or auxiliary tool changes

Examples:
```
feat: add support for sparse vector queries
fix: handle missing collection error in search
docs: update README with new examples
```

## Code Guidelines

### Python Style

- Follow PEP 8 guidelines
- Maximum line length: 100 characters
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Testing Guidelines

- Write tests for all new functionality
- Use pytest fixtures for setup/teardown
- Use `pytest-asyncio` for async tests
- Ensure tests are isolated and don't depend on external state

### MCP Tool Design

When adding new MCP tools:

1. Define input schema in `schemas.py` using Pydantic models
2. Implement the tool in `server.py` with proper annotations
3. Add comprehensive tests in `tests/test_server.py`
4. Update README.md with usage examples

Example tool structure:
```python
@mcp.tool(
    name="tool_name",
    annotations={
        "title": "Human Readable Title",
        "readOnlyHint": True/False,
        "destructiveHint": True/False,
        "idempotentHint": True/False,
        "openWorldHint": True/False,
    }
)
async def tool_name(params: ToolInput) -> str:
    """
    Clear description of what the tool does.
    
    Args:
        params (ToolInput): Description of parameters
    
    Returns:
        str: Description of return value
    """
    try:
        # Implementation
        return result
    except Exception as e:
        return handle_error(e)
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages and stack traces

### Feature Requests

For feature requests, please describe:

- The use case
- Proposed solution
- Alternatives considered

## Questions?

Feel free to open an issue for questions or join discussions in existing issues.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

# Development Setup Guide

This guide helps you set up a development environment for Imgdiff.

## Prerequisites

- Python 3.8 or higher
- Git

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mikhalchankasm/Imgdiff.git
cd Imgdiff
```

### 2. Create virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
# Install package with dev dependencies
pip install -e .[dev]

# Or install manually
pip install -r requirements.txt
pip install pytest pytest-cov ruff black mypy pre-commit
```

### 4. Install pre-commit hooks

```bash
pre-commit install
```

This will automatically run code quality checks before each commit.

## Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. These run automatically on `git commit`.

### What gets checked:

- **Black** - Code formatting (100 char line length)
- **Ruff** - Fast Python linter with auto-fix
- **Trailing whitespace** - Removes trailing spaces
- **End of file fixer** - Ensures newline at end of files
- **YAML/JSON/TOML** - Validates config files
- **Large files** - Prevents committing files >1MB
- **Merge conflicts** - Detects unresolved conflicts

### Manual run:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run black --all-files
```

### Bypass hooks (not recommended):

```bash
git commit --no-verify -m "message"
```

## Code Style

### Formatting with Black

```bash
# Format specific files
black imgdiff/core/diff.py

# Format directory
black imgdiff/ tests/

# Check without modifying
black --check imgdiff/
```

**Line length:** 100 characters

### Linting with Ruff

```bash
# Check all files
ruff check imgdiff/ tests/

# Auto-fix issues
ruff check --fix imgdiff/

# Show statistics
ruff check --statistics imgdiff/
```

### Type checking with mypy

```bash
# Check types
mypy imgdiff/

# Strict mode
mypy --strict imgdiff/core/
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=imgdiff --cov-report=html

# Open coverage report
# Windows:
start htmlcov/index.html
# Linux:
xdg-open htmlcov/index.html
# macOS:
open htmlcov/index.html

# Run specific test file
pytest tests/test_core.py

# Run tests matching pattern
pytest tests/ -k "test_diff"

# Verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x
```

## Running the Application

```bash
# GUI
python Imgdiff.py

# Or via package
python -m imgdiff.gui.main
```

## Building EXE (Windows)

```bash
# Using local script
cd local-dev
build_exe.bat

# Manual build
python -m PyInstaller --onefile --windowed --name Imgdiff --icon imgdiff_icon.ico Imgdiff.py
```

## Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes**
   - Edit code
   - Add tests
   - Update documentation

3. **Run checks locally**
   ```bash
   # Format
   black imgdiff/ tests/

   # Lint
   ruff check --fix imgdiff/

   # Type check
   mypy imgdiff/

   # Test
   pytest tests/
   ```

4. **Commit** (pre-commit hooks run automatically)
   ```bash
   git add .
   git commit -m "feat(core): add new feature"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

## Troubleshooting

### Pre-commit hooks fail

If hooks fail:

1. Read the error message
2. Fix the issues manually or run:
   ```bash
   pre-commit run --all-files
   ```
3. Stage fixed files:
   ```bash
   git add .
   ```
4. Commit again

### Black and Ruff conflict

Black has priority - if both complain:

1. Run Black first:
   ```bash
   black imgdiff/
   ```
2. Then Ruff:
   ```bash
   ruff check --fix imgdiff/
   ```

### mypy errors

Common fixes:

```python
# Add type hints
def my_function(x: int) -> str:
    return str(x)

# Ignore specific line
result = some_untyped_lib()  # type: ignore

# Ignore file
# mypy: ignore-errors
```

### Test failures

Debug failing tests:

```bash
# Run with print statements visible
pytest tests/ -s

# Run single test
pytest tests/test_core.py::test_diff_mask_fast

# Drop into debugger on failure
pytest tests/ --pdb
```

## Additional Resources

- [Black documentation](https://black.readthedocs.io/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [pytest documentation](https://docs.pytest.org/)
- [pre-commit documentation](https://pre-commit.com/)
- [mypy documentation](https://mypy.readthedocs.io/)

## Getting Help

- Read [CONTRIBUTING.md](../CONTRIBUTING.md)
- Check [FAQ.md](../FAQ.md)
- Create an issue on [GitHub](https://github.com/mikhalchankasm/Imgdiff/issues)

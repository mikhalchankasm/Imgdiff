# Contributing to Imgdiff

**[Русская версия](#contributing-на-русском)** | **English**

Thank you for your interest in contributing to Imgdiff! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

Be respectful and constructive in all interactions. We're here to build great software together.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Imgdiff.git
   cd Imgdiff
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/my-new-feature
   # or
   git checkout -b fix/my-bug-fix
   ```

## Development Setup

### Install dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -e .[dev]
```

This installs:
- `pytest` — Testing framework
- `pytest-cov` — Coverage reporting
- `ruff` — Fast Python linter
- `black` — Code formatter
- `mypy` — Type checker

### Run the application

```bash
# GUI
python Imgdiff.py

# Or using the package
python -m imgdiff.gui.main
```

## Code Style

We follow PEP 8 with some modifications. Use the provided tools to ensure consistency.

### Formatting

**Black** is used for code formatting:

```bash
black imgdiff/ tests/ Imgdiff.py
```

Line length: **100 characters**

### Linting

**Ruff** is used for linting:

```bash
ruff check imgdiff/ tests/
```

Common rules:
- Use meaningful variable names
- Add docstrings to public functions
- Keep functions focused and small
- Avoid deep nesting (max 3-4 levels)

### Type Hints

Add type hints to all new code:

```python
from typing import Optional, List, Tuple
import numpy as np

def diff_mask_fast(
    a: np.ndarray,
    b: np.ndarray,
    fuzz: int = 10,
    use_lab: bool = True
) -> np.ndarray:
    """
    Compute difference mask between two images.

    Args:
        a: First image (BGR format)
        b: Second image (BGR format)
        fuzz: Threshold for difference detection (0-255)
        use_lab: Use perceptual Lab color space

    Returns:
        Binary mask (0/255) where differences are detected
    """
    ...
```

Check types with mypy:

```bash
mypy imgdiff/
```

## Testing

### Run tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=imgdiff --cov-report=html

# Specific test file
pytest tests/test_core.py

# Verbose output
pytest tests/ -v
```

### Write tests

Add tests for new functionality in `tests/`:

```python
import pytest
from imgdiff import diff_mask_fast
import numpy as np

def test_diff_mask_identical_images():
    """Test that identical images produce empty mask."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = diff_mask_fast(img, img, fuzz=10)
    assert np.count_nonzero(mask) == 0

def test_diff_mask_different_images():
    """Test that different images produce non-empty mask."""
    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = np.ones((100, 100, 3), dtype=np.uint8) * 255
    mask = diff_mask_fast(img_a, img_b, fuzz=10)
    assert np.count_nonzero(mask) > 0
```

## Submitting Changes

### Before submitting

1. **Run all checks**:
   ```bash
   # Format code
   black imgdiff/ tests/

   # Lint
   ruff check imgdiff/ tests/

   # Type check
   mypy imgdiff/

   # Run tests
   pytest tests/ --cov=imgdiff
   ```

2. **Update documentation** if needed (README, docstrings)

3. **Add tests** for new functionality

### Commit messages

Use clear, descriptive commit messages:

```
feat(core): add WebP format support

- Add WebP reading with OpenCV
- Add WebP writing with quality parameter
- Update file filter to include .webp extension

Closes #123
```

Prefix types:
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `style:` — Code style (formatting, no logic change)
- `refactor:` — Code refactoring
- `test:` — Adding/updating tests
- `chore:` — Maintenance tasks

### Create Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```

2. **Open a Pull Request** on GitHub

3. **Fill out the PR template**:
   - Description of changes
   - Related issues (if any)
   - Screenshots (for UI changes)
   - Testing performed

4. **Wait for review** — maintainers will review and provide feedback

## Reporting Bugs

Create an issue on [GitHub Issues](https://github.com/mikhalchankasm/Imgdiff/issues) with:

- **Clear title** — Describe the issue briefly
- **Steps to reproduce** — How to trigger the bug
- **Expected behavior** — What should happen
- **Actual behavior** — What actually happens
- **Environment**:
  - OS and version (Windows 10, Ubuntu 22.04, etc.)
  - Python version (`python --version`)
  - Imgdiff version
  - Relevant dependencies (`pip list`)
- **Screenshots/logs** — If applicable

## Feature Requests

Create an issue with:

- **Use case** — Why is this feature needed?
- **Proposed solution** — How should it work?
- **Alternatives** — Other approaches considered
- **Additional context** — Examples, mockups, etc.

---

## Contributing (на русском)

Спасибо за интерес к участию в развитии Imgdiff!

### Быстрый старт

1. Форкните репозиторий
2. Склонируйте локально
3. Создайте ветку для изменений
4. Установите зависимости: `pip install -e .[dev]`
5. Внесите изменения
6. Запустите проверки:
   ```bash
   black imgdiff/ tests/
   ruff check imgdiff/
   pytest tests/
   ```
7. Создайте Pull Request

### Стиль кода

- Используйте **black** для форматирования (100 символов на строку)
- Проверяйте код с **ruff**
- Добавляйте **type hints** к новому коду
- Пишите **docstrings** для публичных функций
- Добавляйте **тесты** для новой функциональности

### Сообщения коммитов

Формат: `тип(область): краткое описание`

Типы:
- `feat:` — Новая функциональность
- `fix:` — Исправление ошибки
- `docs:` — Изменения в документации
- `refactor:` — Рефакторинг кода
- `test:` — Добавление/обновление тестов

Пример:
```
feat(core): добавлена поддержка формата WebP

- Чтение WebP через OpenCV
- Запись WebP с параметром качества
- Обновлён фильтр файлов для .webp

Closes #123
```

### Вопросы?

Создайте issue на GitHub или напишите в обсуждениях.

---

Thank you for contributing to Imgdiff! 🎉

"""
Конфигурация pytest
"""
import pytest


def pytest_configure(config):
    """Регистрируем маркеры"""
    config.addinivalue_line(
        "markers", "benchmark: бенчмарк-тесты производительности"
    )


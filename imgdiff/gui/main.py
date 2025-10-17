"""
Обёртка для запуска GUI из нового пакета
"""
import sys
import os

# Добавляем корневую директорию в путь для импорта старого Imgdiff.py
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def main():
    """Точка входа для GUI"""
    # Импортируем и запускаем существующий Imgdiff.py
    import Imgdiff
    # Imgdiff.py уже содержит логику запуска


if __name__ == "__main__":
    main()


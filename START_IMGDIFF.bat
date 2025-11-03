@echo off
REM Запуск Imgdiff - GUI приложение для сравнения изображений
REM Двойной клик по этому файлу для запуска приложения

cd /d "%~dp0"

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Запускаем через pythonw (без консоли)
start "" pythonw Imgdiff.py

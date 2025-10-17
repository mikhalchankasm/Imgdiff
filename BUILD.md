# Инструкции по сборке Imgdiff

## Разработка

### Установка в режиме разработки
```bash
pip install -e ".[dev]"
```

### Запуск тестов
```bash
# Все тесты
pytest tests/

# С покрытием
pytest tests/ --cov=imgdiff --cov-report=html

# Только бенчмарки
pytest tests/ -m benchmark --benchmark-only
```

### Линтеры и форматирование
```bash
# Проверка кода
ruff check imgdiff/

# Форматирование
black imgdiff/ tests/

# Проверка типов
mypy imgdiff/
```

---

## Сборка пакета

### Wheel пакет (для PyPI)
```bash
pip install build
python -m build
```

Результат в `dist/`:
- `imgdiff_compare-2.0.0-py3-none-any.whl`
- `imgdiff_compare-2.0.0.tar.gz`

### Установка локального wheel
```bash
pip install dist/imgdiff_compare-2.0.0-py3-none-any.whl
```

---

## Сборка EXE (Windows)

### PyInstaller (рекомендуется)
```bash
pip install pyinstaller

# Один файл
pyinstaller --onefile --windowed --name Imgdiff --icon imgdiff_icon.ico Imgdiff.py

# Папка (быстрее запуск)
pyinstaller --onedir --windowed --name Imgdiff --icon imgdiff_icon.ico Imgdiff.py
```

### Оптимизированная сборка
```bash
pyinstaller --onefile --windowed ^
  --name Imgdiff ^
  --icon imgdiff_icon.ico ^
  --exclude-module matplotlib ^
  --exclude-module pandas ^
  --exclude-module scipy ^
  --add-data "imgdiff;imgdiff" ^
  Imgdiff.py
```

### Без scikit-image (меньший размер)
```bash
pyinstaller --onefile --windowed ^
  --name Imgdiff ^
  --icon imgdiff_icon.ico ^
  --exclude-module skimage ^
  --exclude-module scipy ^
  Imgdiff.py
```

Результат в `dist/Imgdiff.exe`

---

## Публикация релиза

### GitHub Release
1. Перейти на https://github.com/mikhalchankasm/Imgdiff/releases
2. Нажать "Draft a new release"
3. Выбрать тег `v2.0.0`
4. Заполнить описание из CHANGELOG.md
5. Приложить:
   - `dist/Imgdiff.exe` (Windows)
   - `dist/imgdiff_compare-2.0.0-py3-none-any.whl` (Python пакет)
6. Опубликовать

### PyPI (опционально)
```bash
pip install twine

# TestPyPI (тестовый)
twine upload --repository testpypi dist/*

# PyPI (production)
twine upload dist/*
```

---

## Проверка релиза

### После публикации проверить:
- [ ] README отображается корректно на GitHub
- [ ] Тег v2.0.0 создан
- [ ] Release notes заполнены
- [ ] EXE запускается на чистой Windows системе
- [ ] `pip install imgdiff-compare` работает (если опубликовано в PyPI)
- [ ] Все ссылки в README рабочие
- [ ] Лицензия MIT указана

---

## Troubleshooting

### PyInstaller: ModuleNotFoundError
**Решение:** добавить `--hidden-import module_name`

### EXE слишком большой
**Решение:** исключить ненужные модули:
```bash
--exclude-module matplotlib --exclude-module pandas --exclude-module scipy
```

### scikit-image не работает в EXE
**Решение:** собрать без scikit-image, использовать `use_ssim=False`

### Кириллица в путях
**Решение:** используйте `safe_imread/safe_imwrite` из `imgdiff.core.io`


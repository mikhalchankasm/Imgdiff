# Imgdiff - Высокопроизводительное сравнение изображений

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.5+-red.svg)

Мощный инструмент для визуального сравнения изображений с оптимизированными алгоритмами и интуитивным интерфейсом.

[Возможности](#возможности) • [Установка](#установка) • [Использование](#использование) • [Производительность](#производительность)

</div>

---

## 🎯 Возможности

### Ядро сравнения
- **Перцептуальное сравнение** — алгоритм ΔE в пространстве Lab для точного определения визуальных различий
- **Многомасштабный подход (coarse-to-fine)** — ускорение 5-20× на больших изображениях
- **SSIM по Y-каналу** — структурное сравнение для текстов и UI (опционально)
- **Автовыравнивание** — компенсация небольших сдвигов через ECC (Enhanced Correlation Coefficient)
- **Умная фильтрация** — подавление шумов и выделение только значимых различий

### Интерфейсы
- **GUI (PyQt5)** — полнофункциональный интерфейс с:
  - Интерактивный слайдер для A/B сравнения
  - Режимы overlay, contours, heatmap
  - Фильтрация и поиск файлов
  - Drag & Drop поддержка
  - Пакетная обработка
- **CLI (Typer)** — быстрая обработка из командной строки
- **Python API** — интеграция в свои проекты

### Визуализация
- Двухцветное выделение (добавления/удаления)
- Тепловые карты различий
- Контурное выделение
- Настраиваемая прозрачность и цвета

---

## 📦 Установка

### Базовая установка (GUI + API)
```bash
pip install opencv-python numpy PyQt5
git clone https://github.com/mikhalchankasm/Imgdiff.git
cd Imgdiff
pip install -e .
```

### С CLI поддержкой
```bash
pip install -e ".[cli]"
```

### С SSIM метрикой
```bash
pip install -e ".[ssim]"
```

### Полная установка (всё включено)
```bash
pip install -e ".[all]"
```

### Для разработки
```bash
pip install -e ".[dev]"
```

---

## 🚀 Использование

### GUI
```bash
python Imgdiff.py
# или после установки:
imgdiff-gui
```

**Основной процесс:**
1. Выберите папки с изображениями A и B
2. Настройте параметры во вкладке "Настройки сравнения"
3. Выберите папку для результатов
4. Нажмите "Сравнить"

### CLI

**Сравнение двух изображений:**
```bash
imgdiff image_a.png image_b.png -o diff.png --fuzz 10 --lab
```

**С настройками:**
```bash
imgdiff img1.jpg img2.jpg \
  --output result.png \
  --fuzz 8 \
  --min-area 100 \
  --thickness 3 \
  --mode overlay \
  --color-r 255 --color-g 0 --color-b 0
```

**Пакетная обработка:**
```bash
imgdiff batch ./images_a/ ./images_b/ ./results/ --fuzz 10 --pattern "*.png"
```

### Python API

```python
import cv2
from imgdiff import diff_mask_fast, coarse_to_fine, draw_diff_overlay
from imgdiff.core.io import safe_imread, safe_imwrite

# Загрузка изображений
img_a = safe_imread("image_a.png")
img_b = safe_imread("image_b.png")

# Быстрое сравнение
mask = diff_mask_fast(img_a, img_b, fuzz=10, use_lab=True)

# Многомасштабное сравнение (быстрее для больших изображений)
boxes = coarse_to_fine(img_a, img_b, fuzz=10, scale=0.25)

# Создание overlay
overlay = draw_diff_overlay(img_b, mask, color=(0, 0, 255), alpha=0.6)

# Сохранение
safe_imwrite("result.png", overlay)
```

---

## ⚙️ Параметры сравнения

| Параметр | Описание | Оптимальные значения |
|----------|----------|---------------------|
| **fuzz** | Порог различия | Lab: 5-12, RGB: 10-30 |
| **use_lab** | Перцептуальное Lab пространство | `True` (точнее) |
| **use_coarse** | Многомасштабный подход | `True` (быстрее) |
| **min_area** | Минимальная площадь региона (px²) | 20-100 |
| **thickness** | Толщина линий выделения (px) | 2-5 |
| **scale** | Масштаб грубого прохода | 0.2-0.3 |

---

## ⚡ Производительность

### Оптимизации
- **Векторизованные операции** — NumPy + OpenCV без Python-циклов
- **Coarse-to-fine** — грубая маска на 25% масштабе → точная проверка только ROI
- **Морфология только в маске** — операции только на различиях, не на всём кадре
- **Ранний выход** — детектор "нет отличий" за O(N) на пониженной копии
- **Перцептуальная метрика** — ΔE в Lab вместо сырого RGB

### Бенчмарки (примерные значения)

| Разрешение | Прямое сравнение | Coarse-to-fine | Ускорение |
|------------|------------------|----------------|-----------|
| 1920×1080 | ~120 ms | ~15 ms | 8× |
| 3840×2160 | ~480 ms | ~40 ms | 12× |
| 7680×4320 | ~1900 ms | ~110 ms | 17× |

*Intel i7-10700K, OpenCV 4.8, без GPU*

**Запуск бенчмарков:**
```bash
pytest tests/ -m benchmark --benchmark-only
```

---

## 🎨 Режимы визуализации

### Contours (Контуры)
Контурное выделение различий на исходном изображении
```bash
imgdiff img1.png img2.png --mode contours
```

### Overlay (Наложение)
Цветное полупрозрачное выделение
```bash
imgdiff img1.png img2.png --mode overlay
```

### Heatmap (Тепловая карта)
Интенсивность различий в виде colormap
```bash
imgdiff img1.png img2.png --mode heatmap
```

---

## 🔧 Поддерживаемые форматы

**Входные:** PNG, JPG/JPEG, BMP, TIFF/TIF, WebP  
**Выходные:** PNG, JPG, BMP, TIFF

---

## 🧪 Тестирование

```bash
# Все тесты
pytest tests/

# Только бенчмарки
pytest tests/ -m benchmark

# С покрытием
pytest tests/ --cov=imgdiff --cov-report=html
```

---

## 📋 Требования

**Обязательные:**
- Python 3.8+
- OpenCV (opencv-python) 4.5+
- NumPy 1.20+
- PyQt5 5.15+

**Опциональные:**
- scikit-image 0.19+ (для SSIM метрики)
- Typer 0.9+ (для CLI)
- Rich 13.0+ (для красивого CLI вывода)

---

## 🏗️ Архитектура

```
imgdiff/
├── __init__.py          # Публичный API
├── cli.py               # CLI интерфейс (Typer)
├── core/
│   ├── colors.py        # Цветовые преобразования (Lab, ΔE)
│   ├── diff.py          # Алгоритмы сравнения
│   ├── morph.py         # Морфологические операции
│   ├── overlay.py       # Визуализация результатов
│   └── io.py            # Чтение/запись, кэширование
└── gui/
    └── (работает с core через API)
```

---

## 🤝 Сравнение с аналогами

| Инструмент | Язык | Скорость | Lab/SSIM | GUI | CLI |
|------------|------|----------|----------|-----|-----|
| **imgdiff** (этот) | Python | ⚡⚡⚡ | ✅ | ✅ | ✅ |
| n7olkachev/imgdiff | Go | ⚡⚡⚡⚡ | ❌ | ❌ | ✅ |
| mgedmin/imgdiff | Python | ⚡ | ❌ | ❌ | ✅ |
| odiff | OCaml | ⚡⚡⚡⚡ | ❌ | ❌ | ✅ |

**Преимущества:**
- Единственный с полноценным GUI на PyQt5
- Перцептуальные метрики (Lab, SSIM)
- Python API для интеграции
- Многомасштабный подход (coarse-to-fine)

---

## 📝 Лицензия

MIT License - смотрите [LICENSE](LICENSE) файл

---

## 👤 Автор

**mikhalchankasm**
- GitHub: [@mikhalchankasm](https://github.com/mikhalchankasm)
- Репозиторий: [Imgdiff](https://github.com/mikhalchankasm/Imgdiff)

---

## 🙏 Благодарности

- OpenCV за мощную библиотеку компьютерного зрения
- scikit-image за SSIM реализацию
- PyQt5 за GUI фреймворк
- Typer за удобный CLI

---

## 🐛 Известные проблемы и решения

### scikit-image не работает в PyInstaller EXE
**Решение:** используйте `use_ssim=False` или установите флаг при сборке

### Кириллица в путях
**Решение:** автоматически обрабатывается через `safe_imread/safe_imwrite`

### Большой размер EXE
**Решение:** используйте виртуальное окружение и исключите ненужные зависимости

---

<div align="center">

**⭐ Если проект полезен, поставьте звезду на GitHub! ⭐**

</div>

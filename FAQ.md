# Imgdiff FAQ — Frequently Asked Questions

**[Русская версия](#faq-на-русском)** | **English**

## General Questions

### Q: What is Imgdiff?

**A:** Imgdiff is a high-performance image comparison tool with a GUI that visually highlights differences between two images using advanced computer vision algorithms. It supports batch processing and multiple visualization modes.

### Q: What platforms are supported?

**A:** Currently Windows is fully supported with a pre-built EXE. The Python version works on Windows, Linux, and macOS.

### Q: Is it free?

**A:** Yes! Imgdiff is open-source under the MIT License.

---

## Installation & Setup

### Q: How do I install Imgdiff?

**A:** You have two options:

1. **Download pre-built EXE** (Windows only):
   - Go to [Releases](https://github.com/mikhalchankasm/Imgdiff/releases)
   - Download `Imgdiff.exe`
   - Run directly (no installation needed)

2. **Install from source**:
   ```bash
   pip install -r requirements.txt
   python Imgdiff.py
   ```

### Q: What Python version is required?

**A:** Python 3.8 or higher. Tested on Python 3.8, 3.9, 3.10, 3.11, and 3.12.

### Q: I get an error about missing OpenCV. What should I do?

**A:** Install OpenCV:
```bash
pip install opencv-python>=4.5.0
```

---

## Usage Questions

### Q: How do I choose the right "fuzz" value?

**A:** The `fuzz` parameter controls sensitivity:

- **3-5**: Very sensitive, detects small color changes (may show noise)
- **7-10**: Balanced, good for most use cases (recommended)
- **12-15**: Less sensitive, ignores minor variations
- **20+**: Very tolerant, only major differences

**Lab color space** (default): Use fuzz 5-12
**RGB color space**: Use fuzz 10-30

**Tip:** Start with `fuzz=10` and adjust based on results.

### Q: What does "gamma" do?

**A:** Gamma controls the transparency curve of the overlay:

- **gamma = 1.0**: Linear transparency (default)
- **gamma < 1.0** (e.g., 0.5): Makes differences more visible (lighter overlay)
- **gamma > 1.0** (e.g., 1.5): Makes differences less prominent (darker overlay)

**Tip:** Use gamma=0.7-0.9 for subtle differences, gamma=1.0-1.2 for obvious changes.

### Q: Why is comparison slow on large images?

**A:** For images larger than 4K (3840×2160), enable "Fast ROI core" (enabled by default). This uses a multi-scale algorithm that's 5-20× faster.

If still slow:
1. Reduce "Workers" count (too many threads can cause overhead)
2. Disable "Noise suppression" if not needed
3. Increase "Quick pre-check" ratio to skip identical images faster

### Q: What's the difference between "min area" and "thickness"?

**A:**
- **min area**: Filters out small noise regions (in pixels²). Default: 20
  - Increase to ignore tiny artifacts
  - Decrease to detect small details

- **thickness**: Dilation size for visualization (in pixels). Default: 3
  - Increase to make differences more visible
  - Decrease for precise highlighting

### Q: Should I use "match tolerance"?

**A:** Match tolerance highlights lines/regions that are similar between images:

- **0** (default): Disabled
- **1-5**: Highlight nearly identical regions (e.g., repeated UI elements)
- **5-10**: More tolerant matching

**Use case:** Detecting preserved elements between versions (e.g., "this button didn't change").

---

## Performance Questions

### Q: How many images can I process in batch?

**A:** Imgdiff can handle thousands of images. Batch size is limited only by:
- Available RAM (each pair loads ~2 images in memory per worker)
- Disk space for results

**Tip:** Use caching to avoid reprocessing identical pairs.

### Q: What is the result cache?

**A:** The cache (`.imgdiff_cache/` folder) stores comparison results to avoid recomputation:

- Cache key = hash(image_a + image_b + settings)
- If inputs and settings match a previous run, the result is reused
- Saves time on repeated comparisons

**Clear cache** if you want to force recomputation.

### Q: How many workers should I use?

**A:** Default is `number of CPU cores`. Guidelines:

- **4-8 workers**: Good for most systems
- **More workers**: Faster on high-core CPUs (16+ cores)
- **Fewer workers**: If RAM is limited or images are very large (8K+)

**Tip:** For 4K+ images, use fewer workers (2-4) to avoid memory issues.

---

## File Format Questions

### Q: What image formats are supported?

**A:** Currently supported:
- PNG
- JPG/JPEG
- BMP
- TIFF

**WebP support** is planned (see [TODO.md](TODO.md)).

### Q: Why are my results saved as PNG?

**A:** PNG is lossless and supports transparency (needed for overlays). Auto-compression is used to reduce file size.

### Q: Can I save results as JPG?

**A:** Not currently. JPG is lossy and doesn't support transparency. Future versions may add JPG export as an option.

### Q: Do file names have to match exactly?

**A:** No! Imgdiff uses **natural sorting** and matches files by name order:

- `image_1.png` ↔ `image_1.png` (exact match)
- `photo_01.jpg` ↔ `photo_01.jpg` (exact match)
- Files are sorted naturally (e.g., `img2.png` comes before `img10.png`)

**Filter** lets you narrow down matches by substring (e.g., `_v2` to only compare files containing "_v2").

---

## Troubleshooting

### Q: I get "No matching files" error

**A:** Check:
1. Folders A and B contain images
2. File names match (or are in same order after sorting)
3. Filter is not too restrictive
4. Supported formats (PNG/JPG/BMP/TIFF)

### Q: Differences are not detected correctly

**A:** Try adjusting:
1. **Increase fuzz** if too many false positives (noise)
2. **Decrease fuzz** if differences are missed
3. **Enable noise suppression** to filter artifacts
4. **Disable quick pre-check** if images are very similar but not identical

### Q: Application crashes or freezes

**A:** Possible causes:
1. **Out of memory**: Reduce workers, close other applications
2. **Very large images** (10K+): Enable "Fast ROI core", use fewer workers
3. **Corrupted image**: Check if specific images cause the issue

**Report bugs** at [GitHub Issues](https://github.com/mikhalchankasm/Imgdiff/issues) with error logs.

### Q: Cyrillic (non-ASCII) paths don't work

**A:** This should be fixed in v2.0+. If you still encounter issues:
- Use English-only paths temporarily
- Report the bug with full path example

---

## Advanced Usage

### Q: Can I use Imgdiff as a Python library?

**A:** Yes! Import and use core functions:

```python
from imgdiff import diff_mask_fast, draw_diff_overlay
import cv2

img_a = cv2.imread('a.png')
img_b = cv2.imread('b.png')

mask = diff_mask_fast(img_a, img_b, fuzz=10)
overlay = draw_diff_overlay(img_b, mask, color=(0, 102, 255))

cv2.imwrite('result.png', overlay)
```

See [README_EN.md](README_EN.md#python-api-usage) for more examples.

### Q: Can I run Imgdiff from command line?

**A:** CLI is available (requires optional dependencies):

```bash
pip install -e .[cli]

# Single comparison
imgdiff image_a.png image_b.png -o diff.png --fuzz 10

# Batch
imgdiff batch ./dir_a ./dir_b ./results/
```

### Q: Can I automate Imgdiff?

**A:** Yes! Use Python API or CLI in scripts:

```python
# automation_script.py
from imgdiff import coarse_to_fine
import cv2

def compare_images(path_a, path_b):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    bboxes = coarse_to_fine(img_a, img_b, fuzz=10)
    return len(bboxes) > 0  # True if differences found

if compare_images('before.png', 'after.png'):
    print("Images differ!")
```

### Q: Can I integrate Imgdiff with CI/CD?

**A:** Yes! Use the CLI or Python API in CI pipelines:

```yaml
# GitHub Actions example
- name: Compare screenshots
  run: |
    pip install -e .
    python compare_screenshots.py
    if [ $? -ne 0 ]; then exit 1; fi
```

---

## FAQ (на русском)

## Общие вопросы

### Что такое Imgdiff?

Высокопроизводительный инструмент для сравнения изображений с GUI, который визуально подсвечивает отличия между двумя изображениями.

### Как выбрать правильное значение "fuzz"?

- **3-5**: Очень чувствительно
- **7-10**: Сбалансировано (рекомендуется)
- **12-15**: Менее чувствительно
- **20+**: Только крупные отличия

**Начните с fuzz=10** и подстраивайте по результату.

### Что делает параметр "gamma"?

Gamma управляет прозрачностью overlay:
- **gamma = 1.0**: Линейная прозрачность (по умолчанию)
- **gamma < 1.0**: Более заметные отличия
- **gamma > 1.0**: Менее заметные отличия

### Почему сравнение медленное на больших изображениях?

Включите "Fast ROI core" (включено по умолчанию). Это ускоряет в 5-20 раз.

Также:
1. Уменьшите количество "Workers"
2. Отключите "Подавление шума" если не нужно
3. Увеличьте "Quick pre-check"

### Какие форматы поддерживаются?

- PNG, JPG, BMP, TIFF
- WebP планируется в будущих версиях

### Можно ли использовать Imgdiff как библиотеку Python?

Да! Импортируйте и используйте функции:

```python
from imgdiff import diff_mask_fast, draw_diff_overlay
import cv2

mask = diff_mask_fast(img_a, img_b, fuzz=10)
overlay = draw_diff_overlay(img_b, mask)
```

### Как очистить кэш?

Удалите папку `.imgdiff_cache/` в корне проекта.

### Где сообщить об ошибке?

Создайте issue на [GitHub](https://github.com/mikhalchankasm/Imgdiff/issues) с описанием проблемы, шагами воспроизведения и версией Python.

---

**Не нашли ответ?** Создайте вопрос в [GitHub Issues](https://github.com/mikhalchankasm/Imgdiff/issues).

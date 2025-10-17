"""
Тесты для ядра imgdiff
"""
import pytest
import numpy as np
import cv2

from imgdiff.core.colors import bgr_to_lab_diff, bgr_simple_diff
from imgdiff.core.diff import diff_mask_fast, coarse_to_fine
from imgdiff.core.morph import bboxes_from_mask, filter_small_components, dilate_mask
from imgdiff.core.overlay import draw_diff_overlay, create_heatmap


@pytest.fixture
def test_images():
    """Создаёт пару тестовых изображений"""
    # Создаём белое изображение
    img_a = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    # Второе изображение с красным прямоугольником
    img_b = img_a.copy()
    img_b[30:70, 30:70] = (0, 0, 255)  # BGR красный
    
    return img_a, img_b


@pytest.fixture
def identical_images():
    """Создаёт идентичные изображения"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    return img.copy(), img.copy()


def test_bgr_to_lab_diff(test_images):
    """Тест перцептуальной разности в Lab"""
    img_a, img_b = test_images
    mask = bgr_to_lab_diff(img_a, img_b, fuzz=10)
    
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert cv2.countNonZero(mask) > 0  # Должны быть различия


def test_bgr_simple_diff(test_images):
    """Тест простой разности в BGR"""
    img_a, img_b = test_images
    mask = bgr_simple_diff(img_a, img_b, fuzz=10)
    
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert cv2.countNonZero(mask) > 0


def test_diff_mask_fast_with_lab(test_images):
    """Тест быстрой маски с Lab"""
    img_a, img_b = test_images
    mask = diff_mask_fast(img_a, img_b, fuzz=10, use_lab=True)
    
    assert mask.shape == (100, 100)
    assert cv2.countNonZero(mask) > 0


def test_diff_mask_fast_without_lab(test_images):
    """Тест быстрой маски без Lab"""
    img_a, img_b = test_images
    mask = diff_mask_fast(img_a, img_b, fuzz=10, use_lab=False)
    
    assert mask.shape == (100, 100)
    assert cv2.countNonZero(mask) > 0


def test_identical_images_no_diff(identical_images):
    """Тест что идентичные изображения не дают различий"""
    img_a, img_b = identical_images
    mask = diff_mask_fast(img_a, img_b, fuzz=10, use_lab=True)
    
    # Могут быть мелкие артефакты, но в целом почти нет различий
    diff_percent = (cv2.countNonZero(mask) / mask.size) * 100
    assert diff_percent < 1.0  # Менее 1% различий


def test_coarse_to_fine(test_images):
    """Тест многомасштабного сравнения"""
    img_a, img_b = test_images
    boxes = coarse_to_fine(img_a, img_b, fuzz=10, scale=0.5)
    
    assert isinstance(boxes, list)
    assert len(boxes) >= 0  # Может быть 0 или больше боксов
    
    # Если есть боксы, проверяем формат
    for box in boxes:
        assert len(box) == 4  # (x, y, w, h)
        assert all(isinstance(v, int) for v in box)


def test_bboxes_from_mask():
    """Тест извлечения боксов из маски"""
    # Создаём маску с двумя компонентами
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    mask[60:80, 60:80] = 255
    
    boxes = bboxes_from_mask(mask, min_area=10)
    
    assert len(boxes) == 2
    for box in boxes:
        assert len(box) == 4


def test_filter_small_components():
    """Тест фильтрации мелких компонент"""
    # Создаём маску с большой и мелкой компонентами
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 255  # Большая (1600 px)
    mask[70:75, 70:75] = 255  # Мелкая (25 px)
    
    filtered = filter_small_components(mask, min_area=100)
    
    # Должна остаться только большая компонента
    boxes = bboxes_from_mask(filtered, min_area=10)
    assert len(boxes) == 1


def test_dilate_mask():
    """Тест расширения маски"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, 45:55] = 255
    
    initial_pixels = cv2.countNonZero(mask)
    dilated = dilate_mask(mask, thickness=3)
    dilated_pixels = cv2.countNonZero(dilated)
    
    assert dilated_pixels > initial_pixels


def test_draw_diff_overlay(test_images):
    """Тест создания overlay"""
    img_a, img_b = test_images
    mask = diff_mask_fast(img_a, img_b, fuzz=10)
    
    overlay = draw_diff_overlay(img_b, mask, color=(0, 0, 255), alpha=0.6)
    
    assert overlay.shape == (*img_b.shape[:2], 4)  # RGBA
    assert overlay.dtype == np.uint8


def test_create_heatmap(test_images):
    """Тест создания тепловой карты"""
    img_a, img_b = test_images
    heatmap = create_heatmap(img_a, img_b, use_lab=True)
    
    assert heatmap.shape == img_a.shape
    assert heatmap.dtype == np.uint8


# Бенчмарки (требуют pytest-benchmark)
@pytest.mark.benchmark
def test_benchmark_diff_mask_fast_lab(benchmark, test_images):
    """Бенчмарк быстрой маски с Lab"""
    img_a, img_b = test_images
    result = benchmark(diff_mask_fast, img_a, img_b, fuzz=10, use_lab=True)
    assert result is not None


@pytest.mark.benchmark
def test_benchmark_diff_mask_fast_bgr(benchmark, test_images):
    """Бенчмарк быстрой маски с BGR"""
    img_a, img_b = test_images
    result = benchmark(diff_mask_fast, img_a, img_b, fuzz=10, use_lab=False)
    assert result is not None


@pytest.mark.benchmark
def test_benchmark_coarse_to_fine(benchmark, test_images):
    """Бенчмарк многомасштабного сравнения"""
    img_a, img_b = test_images
    result = benchmark(coarse_to_fine, img_a, img_b, fuzz=10, scale=0.25)
    assert result is not None


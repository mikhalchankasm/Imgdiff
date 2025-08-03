import numpy as np
import cv2
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from core.diff_two_color import diff_two_color

def make_img(shape=(100, 100, 3), color=(255, 255, 255)):
    return np.full(shape, color, dtype=np.uint8)

def test_identical():
    img1 = make_img()
    img2 = make_img()
    overlay, meta = diff_two_color(img1, img2)
    assert meta['diff_pixels'] == 0
    assert meta['same_percent'] == pytest.approx(100.0)

def test_small_change():
    img1 = make_img()
    img2 = make_img()
    img2[10:20, 10:20] = (0, 0, 0)
    overlay, meta = diff_two_color(img1, img2, sens=1.0)
    assert meta['diff_pixels'] > 0
    assert meta['same_percent'] < 100.0

def test_noise():
    rng = np.random.default_rng(42)
    img1 = make_img()
    img2 = img1.copy()
    noise = rng.integers(-5, 5, img1.shape, dtype=np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    overlay, meta = diff_two_color(img1, img2, sens=1.0, blur=3, morph_open=True)
    # Должно быть мало diff-пикселей, если фильтры работают
    assert meta['diff_percent'] < 5.0 
"""
Оптимизированные функции сравнения изображений
"""
import cv2
import numpy as np
from typing import List, Tuple

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

from .colors import bgr_to_lab_diff, bgr_simple_diff
from .morph import bboxes_from_mask


def diff_mask_fast(
    a: np.ndarray,
    b: np.ndarray,
    fuzz: int = 10,
    use_lab: bool = True,
    noise_filter: bool = True
) -> np.ndarray:
    """
    Быстрая векторизованная разность с морфологией только в маске.
    
    :param a: BGR изображение A
    :param b: BGR изображение B
    :param fuzz: порог различия (Lab: 5-12, BGR: 10-30)
    :param use_lab: использовать перцептуальное Lab пространство
    :param noise_filter: применять фильтрацию шума
    :return: бинарная маска различий 0/255
    """
    # 1. Вычисление разности
    if use_lab:
        m = bgr_to_lab_diff(a, b, fuzz=fuzz)
    else:
        m = bgr_simple_diff(a, b, fuzz=fuzz)
    
    # 2. Лёгкое шумоподавление
    if noise_filter:
        m = cv2.medianBlur(m, 3)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    
    return m


def coarse_to_fine(
    a: np.ndarray,
    b: np.ndarray,
    fuzz: int = 10,
    scale: float = 0.25,
    min_area: int = 50,
    use_lab: bool = True
) -> List[Tuple[int, int, int, int]]:
    """
    Многомасштабное сравнение: грубо → точно.
    Ускорение 5-20× на больших изображениях.
    
    :param a: BGR изображение A
    :param b: BGR изображение B
    :param fuzz: порог различия
    :param scale: масштаб грубого прохода (0.2-0.3 оптимально)
    :param min_area: минимальная площадь региона
    :param use_lab: использовать Lab пространство
    :return: список боксов (x, y, w, h) с различиями
    """
    # 1. Грубая маска на уменьшенных копиях
    a_small = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b_small = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    m_small = diff_mask_fast(
        a_small, b_small,
        fuzz=max(3, int(fuzz * scale)),
        use_lab=use_lab
    )
    
    boxes_small = bboxes_from_mask(m_small, min_area=10)
    
    # 2. Пересчёт боксов на full-res и точная проверка только внутри ROI
    boxes = []
    for (xs, ys, ws, hs) in boxes_small:
        x = int(xs / scale)
        y = int(ys / scale)
        w = int(ws / scale)
        h = int(hs / scale)
        
        # Проверяем границы
        h_max, w_max = a.shape[:2]
        if x >= w_max or y >= h_max:
            continue
            
        # Корректируем размеры ROI
        w = min(w, w_max - x)
        h = min(h, h_max - y)
        
        if w <= 0 or h <= 0:
            continue
        
        roi_a = a[y:y+h, x:x+w]
        roi_b = b[y:y+h, x:x+w]
        
        m = diff_mask_fast(roi_a, roi_b, fuzz=fuzz, use_lab=use_lab)
        
        if cv2.countNonZero(m) > 0:
            boxes.append((x, y, w, h))
    
    return boxes


def ssim_mask(
    a: np.ndarray,
    b: np.ndarray,
    win: int = 11,
    thresh: float = 0.85
) -> np.ndarray:
    """
    SSIM-маска по Y-каналу (яркость).
    Только для случаев, когда установлен scikit-image.
    
    :param a: BGR изображение A
    :param b: BGR изображение B
    :param win: размер окна SSIM (7-11)
    :param thresh: порог схожести (0.8-0.9)
    :return: бинарная маска различий 0/255
    """
    if not SSIM_AVAILABLE:
        # Fallback на простую разность
        return diff_mask_fast(a, b, use_lab=False)
    
    # Считаем SSIM по яркости (канал Y)
    y_a = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb)[..., 0]
    y_b = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[..., 0]
    
    _, full = ssim(y_a, y_b, win_size=win, data_range=255, full=True)
    
    # full ∈ [0..1] — карта схожести; маска различий:
    return (full < thresh).astype(np.uint8) * 255


def align_images_ecc(
    a: np.ndarray,
    b: np.ndarray,
    max_iterations: int = 50
) -> np.ndarray:
    """
    Выравнивание изображения B по A с помощью ECC (Enhanced Correlation Coefficient).
    
    :param a: опорное изображение BGR
    :param b: изображение для выравнивания BGR
    :param max_iterations: максимум итераций
    :return: выровненное изображение b
    """
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    
    warp = np.eye(2, 3, dtype=np.float32)  # аффинная матрица
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, 1e-5)
    
    try:
        _, warp = cv2.findTransformECC(a_g, b_g, warp, cv2.MOTION_AFFINE, criteria)
        b_aligned = cv2.warpAffine(b, warp, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR)
        return b_aligned
    except cv2.error:
        # Если выравнивание не удалось, возвращаем оригинал
        return b


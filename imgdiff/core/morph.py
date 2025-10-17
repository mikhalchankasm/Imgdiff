"""
Морфологические операции и работа с компонентами
"""
import cv2
import numpy as np
from typing import List, Tuple


def bboxes_from_mask(m: np.ndarray, min_area: int = 50) -> List[Tuple[int, int, int, int]]:
    """
    Извлекает боксы компонент из бинарной маски.
    Использует connectedComponentsWithStats (быстрее findContours).
    
    :param m: бинарная маска 0/255
    :param min_area: минимальная площадь компоненты
    :return: список (x, y, w, h)
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):  # пропускаем фон (0)
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((x, y, w, h))
    return boxes


def filter_small_components(
    mask: np.ndarray,
    min_area: int = 20
) -> np.ndarray:
    """
    Фильтрует мелкие компоненты из маски (шумоподавление).
    
    :param mask: бинарная маска
    :param min_area: минимальная площадь
    :return: отфильтрованная маска
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_out = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(mask_out, [cnt], -1, 255, -1)
    return mask_out


def dilate_mask(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """
    Расширяет маску для лучшей видимости (толщина линий).
    
    :param mask: бинарная маска
    :param thickness: толщина в пикселях
    :return: расширенная маска
    """
    if thickness <= 1:
        return mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    return cv2.dilate(mask, kernel)


def close_gaps(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Закрывает небольшие разрывы в маске (морфологическое закрытие).
    
    :param mask: бинарная маска
    :param size: размер kernel
    :return: маска с закрытыми разрывами
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


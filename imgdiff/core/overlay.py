"""
Отрисовка результатов сравнения
"""
import cv2
import numpy as np
from typing import Tuple, List


def draw_diff_overlay(
    base_img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.6,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Создаёт overlay с цветным выделением различий.
    
    :param base_img: исходное изображение BGR
    :param mask: бинарная маска различий
    :param color: цвет выделения (BGR)
    :param alpha: прозрачность (0-1)
    :param gamma: экспонента для адаптивной прозрачности
    :return: RGBA overlay
    """
    h, w = base_img.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., :3] = base_img
    overlay[..., 3] = 0
    
    # Применяем цвет к маске
    overlay[mask > 0, :3] = color
    overlay[mask > 0, 3] = int(255 * alpha)
    
    # Alpha-weight по интенсивности
    if gamma != 1.0:
        alpha_map = (mask / 255.0) ** gamma
        overlay[..., 3] = (overlay[..., 3].astype(np.float32) * alpha_map).astype(np.uint8)
    
    return overlay


def draw_two_color_overlay(
    base_img: np.ndarray,
    mask_add: np.ndarray,
    mask_del: np.ndarray,
    add_color: Tuple[int, int, int] = (0, 102, 255),
    del_color: Tuple[int, int, int] = (255, 51, 0),
    alpha: float = 0.6
) -> np.ndarray:
    """
    Двухцветный overlay: добавления и удаления.
    
    :param base_img: исходное изображение BGR
    :param mask_add: маска добавлений
    :param mask_del: маска удалений
    :param add_color: цвет добавлений (BGR)
    :param del_color: цвет удалений (BGR)
    :param alpha: прозрачность
    :return: RGBA overlay
    """
    h, w = base_img.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., :3] = base_img
    overlay[..., 3] = 0
    
    # Применяем цвета
    overlay[mask_add > 0, :3] = add_color
    overlay[mask_add > 0, 3] = int(255 * alpha)
    
    overlay[mask_del > 0, :3] = del_color
    overlay[mask_del > 0, 3] = int(255 * alpha)
    
    return overlay


def create_heatmap(
    a: np.ndarray,
    b: np.ndarray,
    use_lab: bool = True
) -> np.ndarray:
    """
    Создаёт тепловую карту различий (heatmap).
    
    :param a: изображение A (BGR)
    :param b: изображение B (BGR)
    :param use_lab: использовать Lab пространство
    :return: цветная heatmap (BGR)
    """
    if use_lab:
        a_lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
        b_lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
        d = cv2.absdiff(a_lab, b_lab)
        diff = np.sqrt(
            d[..., 0].astype(np.float32) ** 2 +
            d[..., 1].astype(np.float32) ** 2 +
            d[..., 2].astype(np.float32) ** 2
        )
    else:
        d = cv2.absdiff(a, b)
        diff = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Нормализация
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Применяем colormap
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    
    return heatmap


def draw_contours_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Рисует контуры маски на изображении.
    
    :param img: исходное изображение BGR
    :param mask: бинарная маска
    :param color: цвет контуров (BGR)
    :param thickness: толщина линий
    :return: изображение с контурами
    """
    result = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


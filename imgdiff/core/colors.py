"""
Цветовые преобразования и перцептуальные метрики
"""
import cv2
import numpy as np


def bgr_to_lab_diff(a: np.ndarray, b: np.ndarray, fuzz: int = 10) -> np.ndarray:
    """
    Вычисляет перцептуальную разность между изображениями в пространстве Lab.
    Возвращает бинарную маску различий.
    
    :param a: BGR изображение A
    :param b: BGR изображение B
    :param fuzz: порог различия в единицах Lab (5-12 оптимально)
    :return: бинарная маска 0/255
    """
    # Конверсия в Lab (минимум копий)
    a_lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    b_lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
    
    # ΔE*ab ≈ L2 в Lab (быстро и перцептуально корректно)
    d = cv2.absdiff(a_lab, b_lab)
    
    # Нормируем к скалярной "силе различия" без Python-циклов
    diff = np.sqrt(
        d[..., 0].astype(np.float32) ** 2 +
        d[..., 1].astype(np.float32) ** 2 +
        d[..., 2].astype(np.float32) ** 2
    )
    
    # Порог по fuzz в единицах Lab
    m = (diff >= float(fuzz)).astype(np.uint8) * 255
    
    return m


def bgr_simple_diff(a: np.ndarray, b: np.ndarray, fuzz: int = 10) -> np.ndarray:
    """
    Простая разность в BGR (быстрее, но менее перцептуально).
    
    :param a: BGR изображение A
    :param b: BGR изображение B
    :param fuzz: порог различия
    :return: бинарная маска 0/255
    """
    m = cv2.absdiff(a, b)
    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = (m >= fuzz).astype(np.uint8) * 255
    return m


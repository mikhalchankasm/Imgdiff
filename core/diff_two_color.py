import cv2
import numpy as np
from pathlib import Path
import logging
import gc
from skimage.metrics import structural_similarity as ssim
logging.basicConfig(filename='diff.log', filemode='a', format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


def diff_two_color(
    old_img: np.ndarray,
    new_img: np.ndarray,
    sens: float = 1.0,
    blur: int = 3,
    morph_open: bool = True,
    min_area: int = 20,
    kernel: int = 3,
    alpha: float = 0.6,
    gamma: float = 1.0,
    del_color=(255, 51, 0),
    add_color=(0, 102, 255),
    debug: bool = False,
    debug_dir: Path = None,
    use_ssim: bool = False,
    match_tolerance: int = 0,
    match_color=(255, 0, 0),  # Blue in BGR
) -> (np.ndarray, dict):
    """
    Двухцветный overlay-diff с LAB, адаптивным порогом, фильтрацией шума и alpha-weight.
    Добавлена функция обнаружения близко расположенных линий.
    Оптимизирован для уменьшения потребления памяти.

    :param old_img: ndarray BGR старого изображения
    :param new_img: ndarray BGR нового изображения
    :param sens: чувствительность (верхний процент яркости, 1-10)
    :param blur: ядро Gauss (0 = нет)
    :param morph_open: применять ли MORPH_OPEN
    :param min_area: минимальная площадь пятна
    :param kernel: толщина Dilate
    :param alpha: базовая прозрачность
    :param gamma: экспонента для alpha-weight
    :param del_color: BGR цвет для "ушло"
    :param add_color: BGR цвет для "появилось"
    :param debug: сохранять ли debug-изображения
    :param debug_dir: путь для debug-вывода
    :param use_ssim: использовать ли SSIM-карту вместо LAB-diff
    :param match_tolerance: расстояние в пикселях для определения "совпадающих" линий (0 = отключено)
    :param match_color: BGR цвет для "совпадающих" линий
    :return: overlay RGBA, метаданные
    """
    logging.info(f"Запуск diff_two_color: sens={sens}, blur={blur}, morph_open={morph_open}, min_area={min_area}, kernel={kernel}, alpha={alpha}, gamma={gamma}, del_color={del_color}, add_color={add_color}, debug={debug}, debug_dir={debug_dir}, match_tolerance={match_tolerance}")
    
    try:
        assert old_img.shape == new_img.shape, "Размеры изображений не совпадают"
    except AssertionError as e:
        logging.warning(f"Размеры не совпадают: {old_img.shape} vs {new_img.shape}")
        raise
    
    h, w = old_img.shape[:2]
    meta = {}

    if use_ssim:
        # Оптимизированная SSIM обработка
        gray_old = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
        gray_new = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        score, diff_map = ssim(gray_old, gray_new, full=True)
        diff_map = (diff_map * 255).astype("uint8")
        gray_add = diff_map
        gray_del = np.zeros_like(diff_map)
        
        # Освобождаем память
        del gray_old, gray_new, score
    else:
        # 1. LAB-конверт (оптимизированный)
        old_lab = cv2.cvtColor(old_img, cv2.COLOR_BGR2LAB)
        new_lab = cv2.cvtColor(new_img, cv2.COLOR_BGR2LAB)
        
        # 2. Знаковые разницы
        diff_add = cv2.subtract(new_lab, old_lab)  # появилось
        diff_del = cv2.subtract(old_lab, new_lab)  # исчезло
        
        # Освобождаем память LAB
        del old_lab, new_lab
        
        # 3. Серый + блюр
        gray_add = cv2.cvtColor(diff_add, cv2.COLOR_BGR2GRAY)
        gray_del = cv2.cvtColor(diff_del, cv2.COLOR_BGR2GRAY)
        
        # Освобождаем память diff
        del diff_add, diff_del
        
        if blur > 0:
            gray_add = cv2.GaussianBlur(gray_add, (blur, blur), 0)
            gray_del = cv2.GaussianBlur(gray_del, (blur, blur), 0)

    # 4. Адаптивный порог
    thr_add = np.percentile(gray_add, 100 - sens)
    thr_del = np.percentile(gray_del, 100 - sens)
    _, mask_add = cv2.threshold(gray_add, thr_add, 255, cv2.THRESH_BINARY)
    _, mask_del = cv2.threshold(gray_del, thr_del, 255, cv2.THRESH_BINARY)

    # 5. Morph open (отключен для предотвращения "дыр" в объектах)
    # if morph_open:
    #     kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     mask_add = cv2.morphologyEx(mask_add, cv2.MORPH_OPEN, kernel_open)
    #     mask_del = cv2.morphologyEx(mask_del, cv2.MORPH_OPEN, kernel_open)

    # 6. Min-area filter (оптимизированный)
    def filter_small_optimized(mask, min_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_out = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                # Возвращаем заливку для сохранения качества текста
                cv2.drawContours(mask_out, [cnt], -1, (255,), -1)
        return mask_out
    
    mask_add = filter_small_optimized(mask_add, min_area)
    mask_del = filter_small_optimized(mask_del, min_area)

    # 7. Dilate (толщина) - равномерная для всех типов линий
    if kernel > 1:
        # Используем одинаковый kernel для всех типов линий
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        mask_add = cv2.dilate(mask_add, k)
        mask_del = cv2.dilate(mask_del, k)
        
        # Дополнительный dilate для более сплошных линий
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_add = cv2.dilate(mask_add, k2)
        mask_del = cv2.dilate(mask_del, k2)

    # 7.5. Улучшенная очистка "рваных" областей (отключена для предотвращения "дыр")
    # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # mask_add = cv2.morphologyEx(mask_add, cv2.MORPH_CLOSE, kernel_close)
    # mask_del = cv2.morphologyEx(mask_del, cv2.MORPH_CLOSE, kernel_close)
    
    # kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # mask_add = cv2.morphologyEx(mask_add, cv2.MORPH_OPEN, kernel_clean)
    # mask_del = cv2.morphologyEx(mask_del, cv2.MORPH_OPEN, kernel_clean)

    # 8. Обнаружение близко расположенных линий (оптимизированное)
    mask_matched = np.zeros_like(mask_add)
    if match_tolerance > 0:
        kernel_match = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (match_tolerance * 2 + 1, match_tolerance * 2 + 1))
        mask_add_dilated = cv2.dilate(mask_add, kernel_match)
        mask_del_dilated = cv2.dilate(mask_del, kernel_match)
        
        mask_matched = cv2.bitwise_and(mask_add_dilated, mask_del_dilated)
        
        # Убираем совпадающие области из исходных масок
        mask_add = cv2.bitwise_and(mask_add, cv2.bitwise_not(mask_matched))
        mask_del = cv2.bitwise_and(mask_del, cv2.bitwise_not(mask_matched))
        
        # Освобождаем память
        del mask_add_dilated, mask_del_dilated

    # 9. Генерация цветного слоя (оптимизированная)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[..., :3] = new_img
    overlay[..., 3] = 0
    
    # Применяем цвета пакетно с одинаковой прозрачностью для всех типов
    overlay[mask_add > 0, :3] = add_color
    overlay[mask_add > 0, 3] = 255  # Возвращаем полную непрозрачность
    overlay[mask_del > 0, :3] = del_color
    overlay[mask_del > 0, 3] = 255  # Возвращаем полную непрозрачность
    overlay[mask_matched > 0, :3] = match_color
    overlay[mask_matched > 0, 3] = 255  # Возвращаем полную непрозрачность

    # 10. Alpha-weight по силе (возвращаем для лучшего качества)
    if gamma != 1.0:
        alpha_map = np.maximum(gray_add, gray_del) / 255.0
        alpha_map = np.power(alpha_map, gamma)
        overlay[..., 3] = (overlay[..., 3].astype(np.float32) * alpha_map).astype(np.uint8)
        del alpha_map

    # 11. Метрики (оптимизированные) с защитой от деления на ноль
    mask_same = np.logical_not((mask_add > 0) | (mask_del > 0) | (mask_matched > 0))
    total_pixels = mask_same.size
    same_pixels = np.count_nonzero(mask_same)
    diff_pixels = total_pixels - same_pixels
    matched_pixels = np.count_nonzero(mask_matched)
    
    # Защита от деления на ноль
    if total_pixels > 0:
        meta['same_percent'] = same_pixels / total_pixels * 100
        meta['diff_percent'] = diff_pixels / total_pixels * 100
        meta['matched_percent'] = matched_pixels / total_pixels * 100
    else:
        # Fallback значения для пустых изображений
        meta['same_percent'] = 100.0
        meta['diff_percent'] = 0.0
        meta['matched_percent'] = 0.0
    
    meta['diff_pixels'] = int(diff_pixels)
    meta['matched_pixels'] = int(matched_pixels)
    meta['total_pixels'] = int(total_pixels)

    # 12. Debug-вывод (оптимизированный)
    if debug and debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "mask_add.png"), mask_add)
        cv2.imwrite(str(debug_dir / "mask_del.png"), mask_del)
        cv2.imwrite(str(debug_dir / "mask_matched.png"), mask_matched)
        cv2.imwrite(str(debug_dir / "alpha.png"), overlay[..., 3])
        cv2.imwrite(str(debug_dir / "overlay_final.png"), cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR))

    # Освобождаем память
    del gray_add, gray_del, mask_add, mask_del, mask_matched, mask_same
    
    # Принудительная сборка мусора для больших изображений
    if h * w > 1000000:  # Если изображение больше 1MP
        gc.collect()
    
    logging.info(f"Результат: diff_pixels={meta['diff_pixels']}, matched_pixels={meta['matched_pixels']}, same_percent={meta['same_percent']:.2f}, diff_percent={meta['diff_percent']:.2f}, matched_percent={meta['matched_percent']:.2f}")
    return overlay, meta 
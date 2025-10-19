import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter
import cv2
import numpy as np


@dataclass
class AlignmentSettings:
    """Настройки смещения для пары изображений"""
    file_a: str
    file_b: str
    offset_x: int = 0
    offset_y: int = 0
    moving_image: str = "B"  # "A" или "B" - какое изображение смещается
    created_at: str = ""
    updated_at: str = ""


class ImageAlignmentManager:
    """Менеджер для работы со смещением изображений"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.settings_file = self.output_dir / "alignment_settings.json"
        self.settings: Dict[str, AlignmentSettings] = {}
        if output_dir:  # Загружаем настройки только если указана папка
            self.load_settings()
    
    def _get_pair_key(self, file_a: str, file_b: str) -> str:
        """Создает уникальный ключ для пары изображений"""
        return f"{Path(file_a).name}__vs__{Path(file_b).name}"
    
    def load_settings(self):
        """Загружает настройки смещения из JSON файла"""
        if not self.settings_file.exists():
            return
        
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, settings_dict in data.items():
                    self.settings[key] = AlignmentSettings(**settings_dict)
        except Exception as e:
            print(f"Ошибка загрузки настроек смещения: {e}")
    
    def save_settings(self):
        """Сохраняет настройки смещения в JSON файл"""
        if not self.output_dir or str(self.output_dir) == ".":
            return  # Не сохраняем если папка не указана
        
        try:
            # Создаем папку если она не существует
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            data = {key: asdict(settings) for key, settings in self.settings.items()}
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Ошибка сохранения настроек смещения: {e}")
    
    def get_alignment(self, file_a: str, file_b: str) -> AlignmentSettings:
        """Получает настройки смещения для пары изображений"""
        key = self._get_pair_key(file_a, file_b)
        if key not in self.settings:
            # Создаем новые настройки по умолчанию
            now = datetime.now().isoformat()
            self.settings[key] = AlignmentSettings(
                file_a=file_a,
                file_b=file_b,
                created_at=now,
                updated_at=now
            )
        return self.settings[key]
    
    def update_alignment(self, file_a: str, file_b: str, offset_x: int, offset_y: int, 
                        moving_image: str = "B"):
        """Обновляет настройки смещения для пары изображений"""
        now = datetime.now().isoformat()
        
        key = self._get_pair_key(file_a, file_b)
        if key in self.settings:
            settings = self.settings[key]
            settings.offset_x = offset_x
            settings.offset_y = offset_y
            settings.moving_image = moving_image
            settings.updated_at = now
        else:
            self.settings[key] = AlignmentSettings(
                file_a=file_a,
                file_b=file_b,
                offset_x=offset_x,
                offset_y=offset_y,
                moving_image=moving_image,
                created_at=now,
                updated_at=now
            )
        
        if self.output_dir and str(self.output_dir) != ".":
            self.save_settings()
    
    def apply_alignment_to_pixmaps(self, pixmap_a: QPixmap, pixmap_b: QPixmap, 
                                  file_a: str, file_b: str) -> Tuple[QPixmap, QPixmap]:
        """Применяет смещение к QPixmap изображениям"""
        settings = self.get_alignment(file_a, file_b)
        
        if settings.offset_x == 0 and settings.offset_y == 0:
            return pixmap_a, pixmap_b
        
        # Создаем новые QPixmap с учетом смещения
        if settings.moving_image == "A":
            # Смещаем изображение A
            result_a = self._create_offset_pixmap(pixmap_a, settings.offset_x, settings.offset_y)
            result_b = pixmap_b
        else:
            # Смещаем изображение B (по умолчанию)
            result_a = pixmap_a
            result_b = self._create_offset_pixmap(pixmap_b, settings.offset_x, settings.offset_y)
        
        return result_a, result_b
    
    def _create_offset_pixmap(self, pixmap: QPixmap, offset_x: int, offset_y: int) -> QPixmap:
        """Создает новое QPixmap со смещением"""
        if offset_x == 0 and offset_y == 0:
            return pixmap
        
        # Создаем новое изображение с увеличенным размером для размещения смещенного изображения
        new_width = pixmap.width() + abs(offset_x)
        new_height = pixmap.height() + abs(offset_y)
        
        # Создаем новое QPixmap с белым фоном
        result = QPixmap(new_width, new_height)
        result.fill()
        
        # Рисуем смещенное изображение
        painter = QPainter(result)
        painter.drawPixmap(
            max(0, offset_x), 
            max(0, offset_y), 
            pixmap
        )
        painter.end()
        
        return result
    
    def calculate_alignment_from_points(self, file_a: str, file_b: str, 
                                      points_a: List[Tuple[int, int]], 
                                      points_b: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Рассчитывает смещение на основе соответствующих точек на изображениях
        
        Args:
            file_a: путь к первому изображению
            file_b: путь ко второму изображению  
            points_a: список точек (x, y) на первом изображении
            points_b: список точек (x, y) на втором изображении
            
        Returns:
            Tuple[int, int]: (offset_x, offset_y)
        """
        if len(points_a) != len(points_b) or len(points_a) < 2:
            raise ValueError("Необходимо минимум 2 соответствующие точки")
        
        # Рассчитываем среднее смещение по всем точкам
        offsets = []
        for (x1, y1), (x2, y2) in zip(points_a, points_b):
            offset_x = x1 - x2
            offset_y = y1 - y2
            offsets.append((offset_x, offset_y))
        
        # Вычисляем медианное смещение для устойчивости к выбросам
        offset_x = int(np.median([o[0] for o in offsets]))
        offset_y = int(np.median([o[1] for o in offsets]))
        
        return offset_x, offset_y
    
    def auto_align_images(self, file_a: str, file_b: str, 
                         moving_image: str = "B") -> Tuple[int, int]:
        """
        Автоматическое выравнивание изображений с помощью OpenCV
        
        Args:
            file_a: путь к первому изображению
            file_b: путь ко второму изображению
            moving_image: какое изображение смещать ("A" или "B")
            
        Returns:
            Tuple[int, int]: (offset_x, offset_y)
        """
        # Загружаем изображения
        img_a = cv2.imread(file_a, cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(file_b, cv2.IMREAD_GRAYSCALE)
        
        if img_a is None or img_b is None:
            raise ValueError("Не удалось загрузить одно из изображений")
        
        # Используем ORB детектор для поиска ключевых точек
        orb = cv2.ORB_create(nfeatures=1000)
        
        # Находим ключевые точки и дескрипторы
        kp_a, des_a = orb.detectAndCompute(img_a, None)
        kp_b, des_b = orb.detectAndCompute(img_b, None)
        
        if des_a is None or des_b is None:
            raise ValueError("Не удалось найти ключевые точки")
        
        # Создаем матчер
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_a, des_b)
        
        # Сортируем по расстоянию
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            raise ValueError("Недостаточно совпадений для выравнивания")
        
        # Берем лучшие совпадения
        good_matches = matches[:min(20, len(matches))]
        
        # Извлекаем координаты соответствующих точек
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Находим матрицу трансформации
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            raise ValueError("Не удалось найти матрицу трансформации")
        
        # Извлекаем смещение из матрицы трансформации
        offset_x = int(M[0, 2])
        offset_y = int(M[1, 2])
        
        # Если смещаем изображение A, инвертируем смещение
        if moving_image == "A":
            offset_x = -offset_x
            offset_y = -offset_y
        
        return offset_x, offset_y
    
    def clear_alignment(self, file_a: str, file_b: str):
        """Очищает настройки смещения для пары изображений"""
        key = self._get_pair_key(file_a, file_b)
        if key in self.settings:
            del self.settings[key]
            if self.output_dir and str(self.output_dir) != ".":
                self.save_settings()
    
    def get_all_alignments(self) -> Dict[str, AlignmentSettings]:
        """Возвращает все сохраненные настройки смещения"""
        return self.settings.copy() 


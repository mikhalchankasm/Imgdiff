"""
Чтение/запись изображений и кэширование
"""
import cv2
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any


def safe_imread(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Безопасное чтение изображения с поддержкой кириллицы в путях.
    
    :param path: путь к файлу
    :param flags: флаги cv2.imread
    :return: изображение или None
    """
    try:
        # Сначала пробуем через numpy (поддержка кириллицы)
        with open(path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, flags)
            return img
    except Exception:
        # Fallback к обычному cv2.imread
        try:
            return cv2.imread(str(path), flags)
        except Exception:
            return None


def safe_imwrite(path: str, img: np.ndarray) -> bool:
    """
    Безопасная запись изображения с поддержкой кириллицы.
    
    :param path: путь к файлу
    :param img: изображение
    :return: успешность операции
    """
    try:
        # Создаём директорию если нужно
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Кодируем в буфер
        ext = Path(path).suffix.lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            ext = '.png'
        
        success, buffer = cv2.imencode(ext, img)
        if success:
            with open(path, 'wb') as f:
                f.write(buffer)
            return True
        return False
    except Exception:
        return False


def compute_image_hash(img: np.ndarray) -> str:
    """
    Вычисляет SHA256 хэш изображения.
    
    :param img: изображение
    :return: hex строка хэша
    """
    return hashlib.sha256(img.tobytes()).hexdigest()


def compute_file_hash(path: str) -> str:
    """
    Вычисляет SHA256 хэш файла.
    
    :param path: путь к файлу
    :return: hex строка хэша
    """
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_settings_hash(settings: Dict[str, Any]) -> str:
    """
    Вычисляет хэш настроек для кэширования.
    
    :param settings: словарь настроек
    :return: hex строка хэша
    """
    settings_str = json.dumps(settings, sort_keys=True)
    return hashlib.sha256(settings_str.encode()).hexdigest()


class ResultCache:
    """
    Кэш результатов сравнения для избежания повторных вычислений.
    """
    
    def __init__(self, cache_dir: str = ".imgdiff_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Загружает индекс кэша"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_index(self):
        """Сохраняет индекс кэша"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception:
            pass
    
    def get_cache_key(self, img_a_hash: str, img_b_hash: str, settings_hash: str) -> str:
        """Формирует ключ кэша"""
        combined = f"{img_a_hash}_{img_b_hash}_{settings_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Dict]:
        """Получает результат из кэша"""
        if cache_key in self.index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        return None
    
    def put(self, cache_key: str, result: Dict):
        """Сохраняет результат в кэш"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            self.index[cache_key] = {
                "file": str(cache_file),
                "timestamp": str(Path(cache_file).stat().st_mtime)
            }
            self._save_index()
        except Exception:
            pass
    
    def clear(self):
        """Очищает кэш"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._save_index()


def resize_for_preview(img: np.ndarray, max_size: int = 2000) -> np.ndarray:
    """
    Изменяет размер изображения для превью (если слишком большое).
    
    :param img: исходное изображение
    :param max_size: максимальный размер по любой стороне
    :return: изменённое изображение или оригинал
    """
    h, w = img.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


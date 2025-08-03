import os
import sys
import shutil
import logging
import cv2
import numpy as np
import re
from pathlib import Path

# Отключаем Qt auto-scaling и HiDPI
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

# flake8: noqa: E402
from PyQt5.QtCore import Qt, QUrl, QSettings, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QDesktopServices, QColor, QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel, QFileDialog,
    QGroupBox, QRadioButton, QMessageBox, QSplitter, QSpinBox, QFormLayout,
    QButtonGroup, QColorDialog, QDoubleSpinBox, QTabWidget,
    QComboBox, QProgressBar, QSizePolicy, QCheckBox
)

from core.diff_two_color import diff_two_color
from core.slider_reveal import SliderReveal
from core.image_alignment import ImageAlignmentManager
from core.alignment_controls import AlignmentControlPanel


MAGICK = "magick"  # по умолчанию
LOGFILE = "imgdiff.log"

# Настройка логирования
logging.basicConfig(
    filename=LOGFILE,
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

MAX_PREVIEW_SIZE = 2000

def natural_sort_key(text):
    """
    Функция для естественной сортировки строк с числами.
    Например: "Page2" < "Page10" < "Page20"
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    pattern = re.split('([0-9]+)', text)
    return [convert(c) for c in pattern]


def cv2_to_qimage(cv_img):
    if cv_img is None:
        return QImage()
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

def safe_cv2_imread(path):
    """Безопасное чтение изображения с поддержкой кириллицы"""
    try:
        # Пробуем сначала через numpy
        import numpy as np
        with open(path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    except Exception:
        # Fallback к обычному cv2.imread
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

def load_pixmap_scaled(path, max_size=MAX_PREVIEW_SIZE):
    img = safe_cv2_imread(path)
    if img is None: return QPixmap()
    h, w = img.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return QPixmap.fromImage(cv2_to_qimage(img))

class DndTableWidget(QTableWidget):
    directory_dropped = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        dir_path = urls[0].toLocalFile()
        if os.path.isdir(dir_path):
            self.directory_dropped.emit(dir_path)

class FilteredTable(QWidget):
    def __init__(self, label, settings_key, parent=None):
        super().__init__(parent)
        self.settings_key = settings_key
        self._layout = QVBoxLayout(self)
        self.dir_btn = QPushButton(label)
        # --- Новый фильтр: QComboBox с историей ---
        filter_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.setEditable(True)
        self.filter_combo.setInsertPolicy(QComboBox.InsertAtTop)
        self.filter_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.filter_combo.setEditable(True)
        self.filter_combo.setMinimumWidth(80)
        self.filter_combo.setMaximumWidth(200)
        self.filter_combo.setToolTip("Фильтр по имени...")
        self.filter_combo.setDuplicatesEnabled(False)
        self.filter_combo.setMaxCount(20)
        self.filter_combo.setCurrentText("")
        self.filter_combo.lineEdit().setPlaceholderText("Фильтр по имени...")
        self.filter_combo.lineEdit().editingFinished.connect(self.add_filter_to_history)
        # Кнопка сброса фильтра
        self.clear_filter_btn = QPushButton('X')
        self.clear_filter_btn.setToolTip("Сбросить фильтр")
        self.clear_filter_btn.setFixedWidth(24)
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        # Кнопка очистки истории
        self.clear_history_btn = QPushButton('C')
        self.clear_history_btn.setToolTip("Очистить историю фильтров")
        self.clear_history_btn.setFixedWidth(24)
        self.clear_history_btn.clicked.connect(self.clear_filter_history)
        # Кнопка обновления директории
        self.refresh_btn = QPushButton('R')
        self.refresh_btn.setToolTip("Обновить содержимое папки")
        self.refresh_btn.setFixedWidth(24)
        self.refresh_btn.clicked.connect(self.refresh_dir)
        # Кнопки сортировки
        self.sort_asc_btn = QPushButton('↑')
        self.sort_asc_btn.setToolTip("Сортировка по возрастанию")
        self.sort_asc_btn.setFixedWidth(24)
        self.sort_asc_btn.clicked.connect(self.sort_ascending)
        self.sort_desc_btn = QPushButton('↓')
        self.sort_desc_btn.setToolTip("Сортировка по убыванию")
        self.sort_desc_btn.setFixedWidth(24)
        self.sort_desc_btn.clicked.connect(self.sort_descending)
        filter_row.addWidget(self.filter_combo)
        filter_row.addWidget(self.clear_filter_btn)
        filter_row.addWidget(self.clear_history_btn)
        filter_row.addWidget(self.refresh_btn)
        filter_row.addWidget(self.sort_asc_btn)
        filter_row.addWidget(self.sort_desc_btn)
        self.table = DndTableWidget(0, 1)
        self.table.directory_dropped.connect(self.load_from_dir)
        self.table.setHorizontalHeaderLabels(["Имя"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(200)
        self._layout.addWidget(self.dir_btn)
        self._layout.addLayout(filter_row)
        self._layout.addWidget(self.table, 1)
        self._layout.addWidget(self.preview)
        self.files = []  # [(name, path)]
        self.filtered = []  # индексы видимых файлов
        self.dir_path = ""
        self.sort_order = "asc"  # asc, desc, none
        self.filter_combo.lineEdit().textChanged.connect(self.apply_filter)
        self.table.currentCellChanged.connect(self.show_preview)
        self.load_filter_history()

    def load_filter_history(self):
        settings = QSettings("imgdiff", "imgdiff_gui")
        history = settings.value(f"{self.settings_key}/filter_history", [])
        if history:
            self.filter_combo.clear()
            self.filter_combo.addItems(history)

    def save_filter_history(self):
        settings = QSettings("imgdiff", "imgdiff_gui")
        items = [self.filter_combo.itemText(i) for i in range(self.filter_combo.count()) if self.filter_combo.itemText(i)]
        settings.setValue(f"{self.settings_key}/filter_history", items)

    def clear_filter(self):
        self.filter_combo.setCurrentText("")
        self.apply_filter()

    def clear_filter_history(self):
        self.filter_combo.clear()
        self.save_filter_history()

    def sort_ascending(self):
        """Сортировка по возрастанию"""
        self.sort_order = "asc"
        self.apply_filter()

    def sort_descending(self):
        """Сортировка по убыванию"""
        self.sort_order = "desc"
        self.apply_filter()

    def refresh_dir(self):
        if self.dir_path and os.path.isdir(self.dir_path):
            exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            files = [
                str(Path(self.dir_path) / f)
                for f in os.listdir(self.dir_path)
                if Path(f).suffix.lower() in exts
            ]
            self.load_files(files, self.dir_path)

    def load_files(self, files, dir_path=None):
        # Сортируем файлы с естественной сортировкой при загрузке
        sorted_files = sorted(files, key=lambda f: natural_sort_key(os.path.basename(f)))
        self.files = [(os.path.basename(f), f) for f in sorted_files]
        if dir_path:
            self.dir_path = dir_path
        self.apply_filter()

    def apply_filter(self):
        text = self.filter_combo.currentText().lower()
        self.table.setRowCount(0)
        self.filtered = []
        
        # Фильтруем файлы
        filtered_files = []
        for idx, (name, path) in enumerate(self.files):
            if text in name.lower():
                filtered_files.append((idx, name, path))
        
        # Сортируем отфильтрованные файлы с естественной сортировкой
        if self.sort_order == "asc":
            filtered_files.sort(key=lambda x: natural_sort_key(x[1]))  # по имени по возрастанию
        elif self.sort_order == "desc":
            filtered_files.sort(key=lambda x: natural_sort_key(x[1]), reverse=True)  # по имени по убыванию
        
        # Добавляем в таблицу
        for idx, name, path in filtered_files:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.filtered.append(idx)
        
        self.show_preview()

    def add_filter_to_history(self):
        text = self.filter_combo.currentText().strip()
        if text and self.filter_combo.findText(text) == -1:
            self.filter_combo.insertItem(0, text)
            self.save_filter_history()

    def selected_files(self):
        rows = set(idx.row() for idx in self.table.selectedIndexes())
        return [self.files[self.filtered[i]][1] for i in rows if i < len(self.filtered)]

    def all_files(self):
        return [self.files[i][1] for i in self.filtered]

    def show_preview(self, *args):
        row = self.table.currentRow()
        if row >= 0 and row < len(self.filtered):
            img_path = self.files[self.filtered[row]][1]
            pix = load_pixmap_scaled(img_path, max_size=400) # Маленькое превью
            self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.preview.clear()

    def save_state(self, settings: QSettings):
        settings.setValue(f"{self.settings_key}/dir", self.dir_path)
        settings.setValue(f"{self.settings_key}/filter", self.filter_combo.currentText())
        settings.setValue(f"{self.settings_key}/sort_order", self.sort_order)
        self.save_filter_history()

    def restore_state(self, settings: QSettings):
        dir_path = settings.value(f"{self.settings_key}/dir", "")
        filter_text = settings.value(f"{self.settings_key}/filter", "")
        sort_order = settings.value(f"{self.settings_key}/sort_order", "asc")
        self.sort_order = sort_order
        self.load_filter_history()
        if dir_path and os.path.isdir(dir_path):
            exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            files = [
                str(Path(dir_path) / f)
                for f in os.listdir(dir_path)
                if Path(f).suffix.lower() in exts
            ]
            self.load_files(files, dir_path)
        self.filter_combo.setCurrentText(filter_text)
        self.apply_filter()

    def load_from_dir(self, dir_path):
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files = [
            str(Path(dir_path) / f)
            for f in os.listdir(dir_path)
            if Path(f).suffix.lower() in exts
        ]
        self.load_files(files, dir_path)

# --- ResultImageView ---
class ResultImageView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.offset = QPoint(0, 0)
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 8.0
        self._drag = False
        self._last_pos = QPoint()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        if not pixmap.isNull():
            self.scale = 1.0
            self.offset = QPoint(0, 0)
        else:
            self.scale = 1.0
            self.offset = QPoint(0, 0)
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        if self.scale == 1.0:
            qp.setRenderHint(QPainter.SmoothPixmapTransform, False)
        else:
            qp.setRenderHint(QPainter.SmoothPixmapTransform, True)
        qp.translate(self.offset)
        qp.scale(self.scale, self.scale)
        if not self.pixmap.isNull():
            qp.drawPixmap(0, 0, self.pixmap)
        qp.end()

    def wheelEvent(self, e):
        # Зум колесом мыши без Ctrl
        angle = e.angleDelta().y()
        factor = 1.2 if angle > 0 else 1/1.2
        old_scale = self.scale
        self.scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        mouse_pos = e.pos()
        rel = mouse_pos - self.offset
        relf = rel * self.scale / old_scale
        self.offset = mouse_pos - QPoint(int(relf.x()), int(relf.y()))
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self._drag = True
            self._last_pos = e.pos()

    def mouseMoveEvent(self, e):
        if self._drag:
            delta = e.pos() - self._last_pos
            self.offset += delta
            self._last_pos = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self._drag = False

    def resizeEvent(self, e):
        self.update()

# --- SliderReveal с overlay-режимом ---
class SliderReveal(QWidget):
    def __init__(self, pixmap_a, pixmap_b, parent=None):
        super().__init__(parent)
        self.pixmap_a = pixmap_a
        self.pixmap_b = pixmap_b
        self.slider_pos = 0.5
        self.overlay_mode = False
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self._drag = False
        self._last_pos = QPoint(0, 0)
        self._drag_mode = False
        self.min_scale = 0.1
        self.max_scale = 10.0
        
        # Кэш для overlay изображения
        self._overlay_cache = None
        self._overlay_cache_key = None
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setPixmaps(self, pixmap_a, pixmap_b):
        self.pixmap_a = pixmap_a
        self.pixmap_b = pixmap_b
        # Инвалидируем кэш при смене изображений
        self._overlay_cache = None
        self._overlay_cache_key = None
        self.update()

    def setOverlayMode(self, enabled):
        self.overlay_mode = enabled
        # Инвалидируем кэш при смене режима
        self._overlay_cache = None
        self._overlay_cache_key = None
        self.update()

    def _generate_overlay_cache(self):
        """Генерирует кэшированное overlay изображение с четкими контурами в исходном разрешении"""
        if self.pixmap_a is None or self.pixmap_b is None:
            return None
            
        # Создаем ключ кэша на основе ID изображений
        cache_key = (id(self.pixmap_a), id(self.pixmap_b))
        if self._overlay_cache is not None and self._overlay_cache_key == cache_key:
            return self._overlay_cache
            
        try:
            # Используем оригинальные изображения в полном разрешении
            img_a = self.pixmap_a.toImage()
            img_b = self.pixmap_b.toImage()
            
            # ОПТИМИЗАЦИЯ: Используем более эффективную конвертацию QImage в numpy
            # Вместо поэлементного доступа используем прямой доступ к байтам
            def qimage_to_np_optimized(qimg):
                # Конвертируем в RGBA8888 для единообразия
                qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
                width = qimg.width()
                height = qimg.height()
                
                # Прямой доступ к байтам изображения (быстрее чем pixel())
                ptr = qimg.bits()
                ptr.setsize(width * height * 4)
                
                # Создаем numpy массив напрямую из байтов
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                return arr.copy()  # Копируем для безопасности
            
            # Конвертируем изображения в numpy массивы
            arr_a = qimage_to_np_optimized(img_a)
            arr_b = qimage_to_np_optimized(img_b)
            
            # Освобождаем память QImage объектов
            del img_a, img_b
            
            # ОПТИМИЗАЦИЯ: Векторизованные операции вместо циклов
            # Создаем маски не-белых пикселей за одну операцию
            # Используем numpy операции для ускорения
            mask_a = np.any(arr_a[:, :, :3] < 250, axis=2)  # Любой канал < 250 = не белый
            mask_b = np.any(arr_b[:, :, :3] < 250, axis=2)
            
            # Создаем результат с белым фоном за одну операцию
            out = np.full_like(arr_a, 255)
            
            # ОПТИМИЗАЦИЯ: Векторизованное применение цветов
            # Применяем цвета пакетно для всех пикселей одновременно
            
            # Красный для изображения A (полупрозрачный)
            out[mask_a] = [255, 0, 0, 120]  # RGBA: красный с alpha=120
            
            # Зеленый только для уникальных пикселей B (не пересекающихся с A)
            only_b = mask_b & ~mask_a  # Логическое И: B И НЕ A
            out[only_b] = [0, 255, 0, 180]  # RGBA: зеленый с alpha=180
            
            # Синий для совпадающих областей (пересечение A и B)
            both = mask_a & mask_b  # Логическое И: A И B
            out[both] = [0, 0, 255, 200]  # RGBA: синий с alpha=200
            
            # ОПТИМИЗАЦИЯ: Освобождаем память промежуточных массивов
            del arr_a, arr_b, mask_a, mask_b, only_b, both
            
            # Конвертируем обратно в QImage напрямую из numpy массива
            overlay = QImage(out.tobytes(), out.shape[1], out.shape[0], out.strides[0], QImage.Format_RGBA8888)
            
            # Освобождаем numpy массив
            del out
            
            # Сохраняем результат в кэш для повторного использования
            self._overlay_cache = overlay
            self._overlay_cache_key = cache_key
            
            return overlay
            
        except Exception as e:
            print(f"Ошибка генерации overlay: {e}")
            return None

    def paintEvent(self, event):
        if not self.pixmap_a or not self.pixmap_b:
            return
            
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.scale != 1.0)
        
        # Применяем трансформации
        qp.translate(self.offset)
        qp.scale(self.scale, self.scale)
        
        if not self.overlay_mode:
            # Обычный режим слайдера
            split_x = int(self.slider_pos * self.pixmap_a.width())
            qp.drawPixmap(0, 0, self.pixmap_a.copy(0, 0, split_x, self.pixmap_a.height()))
            qp.drawPixmap(split_x, 0, self.pixmap_b.copy(split_x, 0, self.pixmap_b.width() - split_x, self.pixmap_b.height()))
            # Линия слайдера в координатах изображения
            qp.setPen(QColor(0, 120, 215, 180))
            qp.drawLine(split_x, 0, split_x, self.pixmap_a.height())
        else:
            # Overlay режим с кэшированием
            overlay = self._generate_overlay_cache()
            if overlay is not None:
                qp.drawImage(0, 0, overlay)
        qp.end()

    def wheelEvent(self, e):
        # Зум колесом мыши без Ctrl
        angle = e.angleDelta().y()
        factor = 1.2 if angle > 0 else 1/1.2
        old_scale = self.scale
        self.scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        mouse_pos = e.pos()
        rel = mouse_pos - self.offset
        relf = rel * self.scale / old_scale
        self.offset = mouse_pos - QPoint(int(relf.x()), int(relf.y()))
        self.update()

    def mousePressEvent(self, e):
        # Средняя кнопка — drag, ЛКМ — слайдер
        if e.button() == Qt.MouseButton.MiddleButton:
            self._drag = True
            self._last_pos = e.pos()
            self._drag_mode = True
        elif e.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = False
            self._drag = False
            self._last_x = e.x()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag and self._drag_mode:
            delta = e.pos() - self._last_pos
            self.offset += delta
            self._last_pos = e.pos()
            self.update()
        elif not self._drag_mode and e.buttons() & Qt.MouseButton.LeftButton and not self.overlay_mode:
            # Преобразуем координаты мыши в координаты изображения с учетом зума и пана
            mouse_x = (e.x() - self.offset.x()) / self.scale
            if self.pixmap_a.width() > 0:
                self.slider_pos = min(max(mouse_x / self.pixmap_a.width(), 0.0), 1.0)
            self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self._drag = False
            self._drag_mode = False
        super().mouseReleaseEvent(e)

    def resizeEvent(self, e):
        self.update()
        super().resizeEvent(e)

class MainWindow(QMainWindow):
    def __init__(self):
        print('init start')
        super().__init__()
        print('step 1')
        self.setWindowTitle("Image Diff UI – Outline")
        self.resize(1400, 800)
        self.settings = QSettings("imgdiff", "imgdiff_gui")
        self.output_dir = self.settings.value("output_dir", "")
        self.dir_a = self.settings.value("dir_a", "")
        self.dir_b = self.settings.value("dir_b", "")
        
        # Инициализируем менеджер смещения изображений
        self.alignment_manager = None  # Будет инициализирован при выборе папки вывода
        self.alignment_control_panel = None  # Будет создан при инициализации UI
        
        print('step 2')
        # --- Радиокнопки сравнения в QGroupBox ---
        self.radio_all = QRadioButton("Сравнить все")
        self.radio_sel = QRadioButton("Сравнить только выделенные")
        self.radio_sel.setChecked(True)
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_all)
        self.radio_group.addButton(self.radio_sel)
        radio_box = QGroupBox("Режим сравнения")
        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.radio_all)
        radio_layout.addWidget(self.radio_sel)
        radio_box.setLayout(radio_layout)
        print('step 3')
        self.compare_btn = QPushButton("Сравнить")
        self.compare_btn.clicked.connect(self.compare)
        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(["Имя", "Статус", ""])
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setColumnHidden(2, True)
        self.result_table.itemDoubleClicked.connect(self.open_result)
        print('step 4')
        self.out_dir_label = QLabel("Папка вывода:")
        self.out_dir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.out_dir_btn = QPushButton("Выбрать папку вывода…")
        self.out_dir_btn.clicked.connect(self.choose_out_dir)
        self.out_dir_refresh_btn = QPushButton('R')
        self.out_dir_refresh_btn.setToolTip("Обновить список результатов")
        self.out_dir_refresh_btn.setFixedWidth(24)
        self.out_dir_refresh_btn.clicked.connect(self.load_results_from_output_dir)
        out_dir_row = QHBoxLayout()
        out_dir_row.addWidget(self.out_dir_btn)
        out_dir_row.addWidget(self.out_dir_refresh_btn)
        print('step 5')
        result_col = QVBoxLayout()
        result_col.addWidget(self.out_dir_label)
        result_col.addLayout(out_dir_row)
        result_col.addWidget(radio_box)
        result_col.addWidget(self.compare_btn)
        result_col.addWidget(QLabel("Результаты:"))
        result_col_w = QWidget()
        result_col_w.setLayout(result_col)
        result_col_w.setMinimumWidth(120)
        result_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        print('step 6')
        # --- Вкладка 1: Настройки сравнения ---
        self.fuzz_spin = QSpinBox()
        self.fuzz_spin.setRange(0, 100)
        self.fuzz_spin.setSuffix(" %")
        self.fuzz_spin.setValue(1)
        self.fuzz_spin.setToolTip("Допуск (процент игнорируемых отличий, как -fuzz в magick)")
        self.fuzz_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.thick_spin = QSpinBox()
        self.thick_spin.setRange(0, 20)
        self.thick_spin.setValue(3)
        self.thick_spin.setToolTip("Толщина линии (px, для Dilate Octagon)")
        self.thick_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.color_btn = QPushButton()
        self.color_btn.setText("Цвет: #FF0000")
        self.color_btn.setStyleSheet("background:#FF0000")
        self.color = QColor("#FF0000")
        self.color_btn.clicked.connect(self.choose_color)
        self.color_btn.setToolTip("Цвет контура отличий (HEX или имя, как в magick)")
        self.color_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        # --- Чекбоксы вместо кнопок ---
        self.noise_chk = QCheckBox("Фильтр шума")
        self.noise_chk.setChecked(True)
        self.noise_chk.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 9999)
        self.min_area_spin.setValue(20)
        self.min_area_spin.setToolTip("Мин. площадь пятна (px)")
        self.min_area_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setToolTip("Экспонента для alpha-weight")
        self.gamma_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.add_color_btn = QPushButton()
        self.add_color_btn.setText("Цвет добавленного: #0066FF")
        self.add_color_btn.setStyleSheet("background:#0066FF")
        self.add_color = QColor("#0066FF")
        self.add_color_btn.clicked.connect(self.choose_add_color)
        self.add_color_btn.setToolTip("Цвет появившегося (HEX или имя)")
        self.add_color_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.debug_chk = QCheckBox("Debug mode")
        self.debug_chk.setChecked(False)
        self.debug_chk.setToolTip("Сохранять маски и alpha в debug/")
        self.debug_chk.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.ssim_chk = QCheckBox("Use SSIM")
        self.ssim_chk.setChecked(False)
        self.ssim_chk.setToolTip("Использовать SSIM-индекс (лучше для текста, медленнее)")
        self.ssim_chk.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        # --- Новый параметр для обнаружения близких линий ---
        self.match_tolerance_spin = QSpinBox()
        self.match_tolerance_spin.setRange(0, 20)
        self.match_tolerance_spin.setValue(0)
        self.match_tolerance_spin.setToolTip("Расстояние в пикселях для определения 'совпадающих' линий (0 = отключено)")
        self.match_tolerance_spin.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.match_color_btn = QPushButton()
        self.match_color_btn.setText("Цвет совпадений: #0000FF")
        self.match_color_btn.setStyleSheet("background:#0000FF; color:white")
        self.match_color = QColor("#0000FF")
        self.match_color_btn.clicked.connect(self.choose_match_color)
        self.match_color_btn.setToolTip("Цвет для 'совпадающих' линий")
        self.match_color_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        param_form = QFormLayout()
        param_form.addRow("Допуск (fuzz)", self.fuzz_spin)
        param_form.addRow("Толщина (px)", self.thick_spin)
        param_form.addRow("Цвет", self.color_btn)
        param_form.addRow("Фильтр шума", self.noise_chk)
        param_form.addRow("Min area", self.min_area_spin)
        param_form.addRow("Gamma", self.gamma_spin)
        param_form.addRow("Цвет добавленного", self.add_color_btn)
        param_form.addRow("Debug", self.debug_chk)
        param_form.addRow("Use SSIM", self.ssim_chk)
        param_form.addRow("Допуск совпадений (px)", self.match_tolerance_spin)
        param_form.addRow("Цвет совпадений", self.match_color_btn)
        param_group = QGroupBox("Параметры Outline")
        param_group.setLayout(param_form)
        param_group.setMaximumWidth(350)
        # --- Пояснения отдельным блоком ---
        param_help = QLabel(
            "<b>Пояснения к параметрам:</b><br>"
            "<b>Допуск (fuzz):</b> Процент пикселей, которые могут отличаться и считаться равными. Чем выше — тем менее чувствительно.<br>"
            "<b>Толщина (px):</b> Толщина линии выделения отличий.<br>"
            "<b>Цвет:</b> Цвет для выделения отличий.<br>"
            "<b>Фильтр шума:</b> Включить фильтрацию мелких шумов.<br>"
            "<b>Min area:</b> Минимальная площадь пятна для выделения.<br>"
            "<b>Gamma:</b> Экспонента для alpha-weight (контраст выделения).<br>"
            "<b>Цвет добавленного:</b> Цвет для новых объектов на изображении.<br>"
            "<b>Debug:</b> Сохранять промежуточные маски и alpha-каналы.<br>"
            "<b>Use SSIM:</b> Использовать SSIM для сравнения (лучше для текста, медленнее).<br>"
            "<b>Допуск совпадений (px):</b> Расстояние в пикселях для определения 'совпадающих' линий. Линии на расстоянии до этого значения считаются совпадающими и окрашиваются в синий цвет.<br>"
            "<b>Цвет совпадений:</b> Цвет для линий, которые считаются совпадающими или почти совпадающими."
        )
        param_help.setWordWrap(True)
        param_help.setMaximumWidth(350)
        settings_layout = QVBoxLayout()
        settings_layout.addWidget(param_group, alignment=Qt.AlignmentFlag.AlignTop)
        settings_layout.addWidget(param_help, alignment=Qt.AlignmentFlag.AlignTop)
        settings_layout.addStretch(1)
        settings_tab = QWidget()
        settings_tab.setLayout(settings_layout)

        # --- Вкладка 2: Сравнение/Слайдер ---
        # Левый столбец: грид A
        print('left_col_w start')
        self.grp_a = FilteredTable("Выбрать папку A…", "A")
        self.grp_a_label = QLabel("Папка A")
        self.grp_a_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_col = QVBoxLayout()
        left_col.addWidget(self.grp_a_label)
        left_col.addWidget(self.grp_a)
        left_col_w = QWidget()
        left_col_w.setLayout(left_col)
        left_col_w.setMinimumWidth(120)
        left_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        print('left_col_w end')
        print('step 7')
        # Средний столбец: грид B
        print('mid_col_w start')
        self.grp_b = FilteredTable("Выбрать папку B…", "B")
        self.grp_b_label = QLabel("Папка B")
        self.grp_b_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mid_col = QVBoxLayout()
        mid_col.addWidget(self.grp_b_label)
        mid_col.addWidget(self.grp_b)
        mid_col_w = QWidget()
        mid_col_w.setLayout(mid_col)
        mid_col_w.setMinimumWidth(120)
        mid_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        print('mid_col_w end')
        print('step 8')
        # Правая колонка: результаты
        self.out_dir_label = QLabel("Папка вывода:")
        self.out_dir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.out_dir_btn = QPushButton("Выбрать папку вывода…")
        self.out_dir_btn.clicked.connect(self.choose_out_dir)
        # Кнопка обновления папки результатов
        self.out_dir_refresh_btn = QPushButton('R')
        self.out_dir_refresh_btn.setToolTip("Обновить список результатов")
        self.out_dir_refresh_btn.setFixedWidth(24)
        self.out_dir_refresh_btn.clicked.connect(self.load_results_from_output_dir)
        out_dir_row = QHBoxLayout()
        out_dir_row.addWidget(self.out_dir_btn)
        out_dir_row.addWidget(self.out_dir_refresh_btn)
        # --- Радиокнопки сравнения в QGroupBox ---
        self.radio_all = QRadioButton("Сравнить все")
        self.radio_sel = QRadioButton("Сравнить только выделенные")
        self.radio_sel.setChecked(True)
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_all)
        self.radio_group.addButton(self.radio_sel)
        radio_box = QGroupBox("Режим сравнения")
        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.radio_all)
        radio_layout.addWidget(self.radio_sel)
        radio_box.setLayout(radio_layout)
        
        # Кнопка сохранения overlay - перемещена под радио-кнопки
        self.save_overlay_btn = QPushButton("Сохранить")
        self.save_overlay_btn.setToolTip("Сохранить overlay для выбранных или всех файлов (в зависимости от выделения)")
        self.save_overlay_btn.clicked.connect(self.save_overlay)
        self.save_overlay_btn.setEnabled(False)  # Включаем только когда overlay активен
        

        
        result_col = QVBoxLayout()
        result_col.addWidget(self.out_dir_label)
        result_col.addLayout(out_dir_row)
        result_col.addWidget(radio_box)
        result_col.addWidget(self.save_overlay_btn)  # Кнопка под радио-кнопками
        result_col.addWidget(QLabel("Результаты:"))
        result_col.addWidget(self.result_table, 1)
        self.open_external_btn = QPushButton("Открыть в системном просмотрщике")
        self.open_external_btn.setToolTip("Открыть выбранный результат в стандартном приложении Windows")
        self.open_external_btn.clicked.connect(self.open_result_external)
        result_col.addWidget(self.open_external_btn)
        self.open_internal_viewer_btn = QPushButton("Открыть в отдельном окне")
        self.open_internal_viewer_btn.setToolTip("Показать выбранный результат в отдельном окне с zoom/drag")
        self.open_internal_viewer_btn.clicked.connect(self.open_result_internal_viewer)
        result_col.addWidget(self.open_internal_viewer_btn)
        result_col_w = QWidget()
        result_col_w.setLayout(result_col)
        result_col_w.setMinimumWidth(120)
        result_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        print('result_col_w end')
        print('step 9')
        print('before splitter')
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        print('after splitter')
        self.splitter.addWidget(left_col_w)
        print('after add left')
        self.splitter.addWidget(mid_col_w)
        print('after add mid')
        self.splitter.addWidget(result_col_w)
        print('after add result')
        self.splitter.setSizes([180, 180, 180])
        print('after setSizes')
        self.splitter.setHandleWidth(4)
        print('after setHandleWidth')
        # --- Слайдер справа ---
        print('before slider_widget')
        self.slider_widget = QWidget()
        self.slider_layout = QVBoxLayout(self.slider_widget)
        print('after slider_widget')
        # --- Панель управления над слайсером ---
        self.slider_control = QHBoxLayout()
        self.overlay_chk = QCheckBox("Overlay")
        self.overlay_chk.setChecked(False)
        self.overlay_chk.setToolTip("Включить режим наложения (A=красный, B=зелёный)")
        self.overlay_chk.stateChanged.connect(self.update_slider_overlay_mode)
        
        # Кнопка "Вписать всё"
        self.fit_to_window_btn = QPushButton("Вписать всё")
        self.fit_to_window_btn.setToolTip("Вписать изображение в окно целиком")
        self.fit_to_window_btn.clicked.connect(self.fit_to_window)
        
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedWidth(32)
        self.prev_btn.clicked.connect(lambda: self.navigate_result(-1))
        self.next_btn = QPushButton(">")
        self.next_btn.setFixedWidth(32)
        self.next_btn.clicked.connect(lambda: self.navigate_result(1))
        self.slider_control.addWidget(self.overlay_chk)
        self.slider_control.addWidget(self.fit_to_window_btn)
        self.slider_control.addStretch(1)
        self.slider_control.addWidget(self.prev_btn)
        self.slider_control.addWidget(self.next_btn)
        self.slider_layout.addLayout(self.slider_control)
        self.slider_header = QHBoxLayout()
        self.label_a = QLabel("A: <не выбрано>")
        self.label_b = QLabel("B: <не выбрано>")
        self.slider_header.addWidget(self.label_a)
        self.slider_header.addStretch(1)
        self.slider_header.addWidget(self.label_b)
        self.slider_layout.addLayout(self.slider_header)
        self.slider_reveal = SliderReveal(QPixmap(), QPixmap())
        self.slider_layout.addWidget(self.slider_reveal, 1)
        self.slider_reveal.setVisible(True)
        
        # Добавляем панель управления смещением
        if self.output_dir:
            self.alignment_manager = ImageAlignmentManager(self.output_dir)
        else:
            self.alignment_manager = ImageAlignmentManager("")
        self.alignment_control_panel = AlignmentControlPanel(self.alignment_manager)
        self.alignment_control_panel.alignment_changed.connect(self.on_alignment_changed)
        self.slider_layout.addWidget(self.alignment_control_panel)
        
        self.slider_widget.setMinimumWidth(600)  # Увеличиваем зону превью в 1.5 раза
        self.slider_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_image_view = ResultImageView()
        self.result_image_view.setVisible(False)
        self.slider_layout.addWidget(self.result_image_view, 1)
        self.current_result_index = 0
        # Удаляем вызов self.update_slider_view_mode()
        # self.update_slider_view_mode()  # Удалить эту строку
        print('after slider setup')
        # --- Главный QSplitter: три колонки + слайдер ---
        print('before main_splitter')
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        print('after main_splitter')
        self.main_splitter.addWidget(self.splitter)
        print('after add splitter')
        self.main_splitter.addWidget(self.slider_widget)
        print('after add slider_widget')
        self.main_splitter.setSizes([540, 900])  # Увеличиваем зону превью в 1.5 раза
        print('after main_splitter setSizes')
        self.main_splitter.setHandleWidth(4)
        print('after main_splitter setHandleWidth')
        # --- Tabs ---
        print('before tabs')
        self.tabs = QTabWidget()
        print('after tabs')
        self.tabs.addTab(settings_tab, "Настройки сравнения")
        print('after add settings_tab')
        main_tab = QWidget()
        main_tab.setLayout(QHBoxLayout())
        main_tab.layout().addWidget(self.main_splitter)
        print('after main_tab layout')
        self.tabs.addTab(main_tab, "Сравнение и Слайдер")
        print('after add main_tab')
        self.setCentralWidget(self.tabs)
        print('after setCentralWidget')
        # --- Современный стиль ---
        self.setStyleSheet('''
            QWidget { background: #f7f7fa; }
            QTableWidget { background: #fff; border: 1px solid #bbb; border-radius: 6px; font-size: 13px; }
            QHeaderView::section { background: #eaeaea; font-weight: bold; border: none; border-bottom: 1px solid #bbb; }
            QPushButton { background: #e0e6f6; border: 1px solid #aab; border-radius: 6px; padding: 4px 10px; font-size: 13px; }
            QPushButton:hover { background: #d0d8f0; }
            QLabel { font-size: 13px; }
            QSplitter::handle { background: #b0b0b0; border: none; }
            QSplitter::handle:hover { background: #0078d7; }
        ''')
        print('after setStyleSheet')
        # --- Status Bar ---
        self.progress_bar = QProgressBar()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        print('after status bar')
        # --- Connections ---
        self.grp_a.dir_btn.clicked.connect(lambda: self.load_files(self.grp_a, 'A'))
        self.grp_b.dir_btn.clicked.connect(lambda: self.load_files(self.grp_b, 'B'))
        self.grp_a.table.itemSelectionChanged.connect(self.update_slider)
        self.grp_b.table.itemSelectionChanged.connect(self.update_slider)
        self.grp_a.table.itemDoubleClicked.connect(self.open_table_image)
        self.grp_b.table.itemDoubleClicked.connect(self.open_table_image)
        self.result_table.itemSelectionChanged.connect(self.on_result_selection_changed)
        
        # Подключаем обновление состояния кнопки сохранения
        self.radio_all.toggled.connect(self.update_save_button_state)
        self.radio_sel.toggled.connect(self.update_save_button_state)
        self.grp_a.table.itemSelectionChanged.connect(self.update_save_button_state)
        self.grp_b.table.itemSelectionChanged.connect(self.update_save_button_state)
        
        print('after connections')
        self.restore_state()
        print('init end')
        
        # Инициализируем состояние кнопки сохранения
        self.update_save_button_state()

    def update_magick_label(self):
        path = self.magick_path or shutil.which(MAGICK) or "<не найден>"
        self.magick_label.setText(f"magick.exe: {path}")

    def choose_magick(self):
        magick_path, _ = QFileDialog.getOpenFileName(self, "Укажите magick.exe", "", "magick.exe (magick.exe)")
        if magick_path:
            self.magick_path = magick_path
            self.settings.setValue("magick_path", magick_path)
            self.update_magick_label()

    def update_out_dir_label(self):
        self.out_dir_label.setText(f"Папка вывода: {self.output_dir or '<не выбрана>'}")

    def choose_out_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Папка для результатов", self.output_dir or "")
        if dir_path:
            self.output_dir = dir_path
            self.settings.setValue("output_dir", dir_path)
            self.update_out_dir_label()
            
            # Инициализируем менеджер смещения для новой папки
            self.alignment_manager = ImageAlignmentManager(dir_path)
            if self.alignment_control_panel:
                self.alignment_control_panel.alignment_manager = self.alignment_manager
            
            self.load_results_from_output_dir()
            self.update_save_button_state()  # Обновляем состояние кнопки сохранения

    def load_results_from_output_dir(self):
        if not self.output_dir or not os.path.isdir(self.output_dir):
            self.result_table.setRowCount(0)
            return
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files = [
            str(Path(self.output_dir) / f)
            for f in sorted(os.listdir(self.output_dir))
            if Path(f).suffix.lower() in exts
        ]
        self.result_table.setRowCount(0)
        for f in files:
            name = os.path.basename(f)
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(name))
            self.result_table.setItem(row, 1, QTableWidgetItem(""))
            self.result_table.setItem(row, 2, QTableWidgetItem(f))
        
        # Очищаем кэш превью при смене папки результатов
        if hasattr(self, '_preview_cache'):
            self._preview_cache.clear()

    def choose_color(self):
        col = QColorDialog.getColor(self.color, self, "Выберите цвет")
        if col.isValid():
            self.color = col
            self.color_btn.setText(f"Цвет: {col.name()}")
            self.color_btn.setStyleSheet(f"background:{col.name()}")

    def choose_add_color(self):
        col = QColorDialog.getColor(self.add_color, self, "Выберите цвет для добавленного")
        if col.isValid():
            self.add_color = col
            self.add_color_btn.setText(f"Цвет добавленного: {col.name()}")
            self.add_color_btn.setStyleSheet(f"background:{col.name()}")

    def choose_match_color(self):
        col = QColorDialog.getColor(self.match_color, self, "Выберите цвет для совпадающих линий")
        if col.isValid():
            self.match_color = col
            self.match_color_btn.setText(f"Цвет совпадений: {col.name()}")
            self.match_color_btn.setStyleSheet(f"background:{col.name()}; color:white")

    def load_files(self, target: FilteredTable, which):
        dir_path = QFileDialog.getExistingDirectory(self, f"Выберите папку с изображениями для {which}", target.dir_path or "")
        if not dir_path:
            return
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files = [
            str(Path(dir_path) / f)
            for f in sorted(os.listdir(dir_path))
            if Path(f).suffix.lower() in exts
        ]
        target.load_files(files, dir_path)
        target.dir_path = dir_path
        target.apply_filter()
        target.show_preview()
        if which == 'A':
            self.dir_a = dir_path
            self.settings.setValue("dir_a", dir_path)
        elif which == 'B':
            self.dir_b = dir_path
            self.settings.setValue("dir_b", dir_path)
        
        # Обновляем состояние кнопки сохранения после загрузки файлов
        self.update_save_button_state()

    def open_result(self, item):
        if item is not None:
            row = item.row()
            cell = self.result_table.item(row, 2)
            if cell is not None:
                path = cell.text()
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def compare(self):
        files_a = self.grp_a.selected_files() if self.radio_sel.isChecked() else self.grp_a.all_files()
        files_b = self.grp_b.selected_files() if self.radio_sel.isChecked() else self.grp_b.all_files()
        if len(files_a) != len(files_b) or not files_a:
            QMessageBox.warning(self, "Несимметричный выбор", "Выделите одинаковое число файлов в обоих списках.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Нет папки вывода", "Сначала выберите папку для результатов.")
            return
        
        # Импортируем gc для управления памятью
        import gc
        
        self.progress_bar.setMaximum(len(files_a))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # Получаем все файлы из папки результатов
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        all_result_files = [
            str(Path(self.output_dir) / f)
            for f in sorted(os.listdir(self.output_dir))
            if Path(f).suffix.lower() in exts
        ]
        self.result_table.setRowCount(0)
        
        # Заполняем грид всеми файлами из папки результатов
        for f in all_result_files:
            name = os.path.basename(f)
            self.result_table.insertRow(self.result_table.rowCount())
            self.result_table.setItem(self.result_table.rowCount()-1, 0, QTableWidgetItem(name))
            self.result_table.setItem(self.result_table.rowCount()-1, 1, QTableWidgetItem(""))
            self.result_table.setItem(self.result_table.rowCount()-1, 2, QTableWidgetItem(f))
        
        # Словарь для быстрого поиска строки по имени
        name_to_row = {self.result_table.item(row, 0).text(): row for row in range(self.result_table.rowCount())}
        
        fuzz = self.fuzz_spin.value()
        thick = self.thick_spin.value()
        hex_color = self.color.name()
        match_tolerance = self.match_tolerance_spin.value()
        match_color = self.match_color
        
        for i, (a, b) in enumerate(zip(files_a, files_b)):
            # Обрабатываем события Qt для отзывчивости UI
            QApplication.processEvents()
            
            out_name = f"{Path(a).stem}__vs__{Path(b).stem}_outline.png"
            out_path = Path(self.output_dir) / out_name
            
            # Обновляем прогресс с информацией о текущем файле
            self.progress_bar.setFormat(f"Обработка: {Path(a).name} vs {Path(b).name}")
            
            try:
                code = self.run_outline(a, b, out_path, fuzz, thick, hex_color, match_tolerance, match_color)
            except FileNotFoundError as e:
                logging.error(f"FileNotFoundError: {e}")
                QMessageBox.critical(self, "Ошибка", f"Не могу открыть файл: {e}")
                self.progress_bar.hide()
                return
            except Exception as e:
                logging.error(f"Exception: {e}")
                status = f"Error: {e}"
                row = name_to_row.get(out_name)
                if row is not None:
                    self.result_table.setItem(row, 1, QTableWidgetItem(status))
                self.progress_bar.setValue(i + 1)
                continue
            
            if code == 0:
                status = "Equal"
            elif code == 1:
                status = "OK"
            else:
                status = "Error"
            
            row = name_to_row.get(out_name)
            if row is not None:
                self.result_table.setItem(row, 1, QTableWidgetItem(status))
            
            self.progress_bar.setValue(i + 1)
            
            # Принудительная сборка мусора каждые 5 файлов для больших изображений
            if (i + 1) % 5 == 0:
                gc.collect()
                QApplication.processEvents()
        
        self.progress_bar.hide()
        self.progress_bar.setFormat("")  # Сбрасываем формат
        
        # Финальная сборка мусора
        gc.collect()
        
        # Подсчитываем статистику результатов
        success_count = 0
        error_count = 0
        equal_count = 0
        for row in range(self.result_table.rowCount()):
            status = self.result_table.item(row, 1).text()
            if status == "OK":
                success_count += 1
            elif status == "Equal":
                equal_count += 1
            elif status.startswith("Error"):
                error_count += 1
        
        # Уведомление о завершении сравнения с детальной статистикой
        message = f"Сравнение завершено!\n\n"
        message += f"Обработано пар изображений: {len(files_a)}\n"
        message += f"Успешно: {success_count}\n"
        message += f"Идентичны: {equal_count}\n"
        message += f"Ошибок: {error_count}\n\n"
        message += f"Результаты сохранены в папку:\n{self.output_dir}"
        
        QMessageBox.information(self, "Сравнение завершено", message)

    def add_result(self, name, status, path):
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        self.result_table.setItem(row, 0, QTableWidgetItem(name))
        self.result_table.setItem(row, 1, QTableWidgetItem(status))
        self.result_table.setItem(row, 2, QTableWidgetItem(path))

    def run_outline(self, left, right, out_path, fuzz, thick, color_hex, match_tolerance, match_color):
        old = safe_cv2_imread(str(left))
        new = safe_cv2_imread(str(right))
        if old is None or new is None:
            raise FileNotFoundError(f"Не могу открыть {left} или {right}")
        
        # Проверяем размер изображений для оптимизации
        old_h, old_w = old.shape[:2]
        new_h, new_w = new.shape[:2]
        
        # Если изображения слишком большие, предупреждаем пользователя
        if old_h > 8000 or old_w > 8000 or new_h > 8000 or new_w > 8000:
            print(f"Предупреждение: Одно из изображений очень большое ({old_w}x{old_h} или {new_w}x{new_h}). Обработка может занять много времени.")
        
        # --- Приводим оба изображения к максимальному размеру с улучшенным качеством ---
        h = max(old.shape[0], new.shape[0])
        w = max(old.shape[1], new.shape[1])
        
        if old.shape[:2] != (h, w):
            # Используем INTER_LANCZOS4 для лучшего качества при увеличении
            old = cv2.resize(old, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if new.shape[:2] != (h, w):
            new = cv2.resize(new, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        sens = fuzz
        blur = 3  # default
        morph_open = self.noise_chk.isChecked()
        min_area = self.min_area_spin.value()
        kernel = thick
        alpha = 0.6  # default
        gamma = self.gamma_spin.value()
        del_color = (self.color.blue(), self.color.green(), self.color.red())
        add_color = (self.add_color.blue(), self.add_color.green(), self.add_color.red())
        match_color_bgr = (match_color.blue(), match_color.green(), match_color.red())
        debug = self.debug_chk.isChecked()
        use_ssim = self.ssim_chk.isChecked()
        debug_dir = Path(self.output_dir) / 'debug' if debug else Path('.')
        
        overlay, meta = diff_two_color(
            old_img=old,
            new_img=new,
            sens=sens,
            blur=blur,
            morph_open=morph_open,
            min_area=min_area,
            kernel=kernel,
            alpha=alpha,
            gamma=gamma,
            del_color=del_color,
            add_color=add_color,
            debug=debug,
            debug_dir=debug_dir,
            use_ssim=use_ssim,
            match_tolerance=match_tolerance,
            match_color=match_color_bgr
        )
        
        # Сохраняем с оптимальным сжатием для уменьшения размера файла
        success = cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not success:
            raise Exception(f"Не удалось сохранить результат в {out_path}")
        
        # Освобождаем память
        del old, new, overlay
        
        return 1 if meta['diff_pixels'] > 0 else 0

    def update_slider(self):
        # Используем новую логику с учетом смещения
        self.update_slider_with_alignment()

    def update_slider_pair(self):
        # этот метод больше не нужен
        pass

    def update_slider_overlay_mode(self):
        self.slider_reveal.setOverlayMode(self.overlay_chk.isChecked())
        # Обновляем состояние кнопки сохранения
        self.update_save_button_state()

    def fit_to_window(self):
        """Вписать изображение в окно целиком"""
        if hasattr(self, 'slider_reveal') and self.slider_reveal.pixmap_a and self.slider_reveal.pixmap_b:
            # Получаем размеры виджета и изображения
            widget_size = self.slider_reveal.size()
            img_width = max(self.slider_reveal.pixmap_a.width(), self.slider_reveal.pixmap_b.width())
            img_height = max(self.slider_reveal.pixmap_a.height(), self.slider_reveal.pixmap_b.height())
            
            # Вычисляем масштаб для вписывания в окно
            scale_x = widget_size.width() / img_width
            scale_y = widget_size.height() / img_height
            scale = min(scale_x, scale_y, 1.0)  # Не увеличиваем больше 1.0
            
            # Применяем масштаб и центрируем
            self.slider_reveal.scale = scale
            self.slider_reveal.offset = QPoint(
                int((widget_size.width() - img_width * scale) // 2),
                int((widget_size.height() - img_height * scale) // 2)
            )
            self.slider_reveal.update()
            
        elif hasattr(self, 'result_image_view') and not self.result_image_view.pixmap.isNull():
            # Получаем размеры виджета и изображения
            widget_size = self.result_image_view.size()
            img_width = self.result_image_view.pixmap.width()
            img_height = self.result_image_view.pixmap.height()
            
            # Вычисляем масштаб для вписывания в окно
            scale_x = widget_size.width() / img_width
            scale_y = widget_size.height() / img_height
            scale = min(scale_x, scale_y, 1.0)  # Не увеличиваем больше 1.0
            
            # Применяем масштаб и центрируем
            self.result_image_view.scale = scale
            self.result_image_view.offset = QPoint(
                int((widget_size.width() - img_width * scale) // 2),
                int((widget_size.height() - img_height * scale) // 2)
            )
            self.result_image_view.update()

    def open_table_image(self, item):
        """Открыть изображение из таблицы A или B в стандартном просмотрщике"""
        if item is not None:
            row = item.row()
            # Определяем из какой таблицы был клик
            sender = self.sender()
            if sender == self.grp_a.table:
                if row < len(self.grp_a.filtered):
                    file_path = self.grp_a.files[self.grp_a.filtered[row]][1]
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))
            elif sender == self.grp_b.table:
                if row < len(self.grp_b.filtered):
                    file_path = self.grp_b.files[self.grp_b.filtered[row]][1]
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))

    def show_result_image_from_selection(self):
        """Оптимизированный метод загрузки превью результата с кэшированием"""
        row = self.result_table.currentRow()
        if row < 0:
            self.result_image_view.setPixmap(QPixmap())
            return
            
        img_path = self.result_table.item(row, 2).text()
        
        # Быстрая проверка существования файла
        if not os.path.isfile(img_path):
            self.result_image_view.setPixmap(QPixmap())
            return
        
        # Проверяем кэш
        if hasattr(self, '_preview_cache') and img_path in self._preview_cache:
            self.result_image_view.setPixmap(self._preview_cache[img_path])
            self.result_image_view.setToolTip(img_path)
            self.current_result_index = row
            return
            
        try:
            # Загружаем изображение напрямую в QPixmap для лучшей производительности
            pix = QPixmap(img_path)
            if not pix.isNull():
                pix.setDevicePixelRatio(1.0)
                # Кэшируем результат
                if not hasattr(self, '_preview_cache'):
                    self._preview_cache = {}
                self._preview_cache[img_path] = pix
                # Ограничиваем размер кэша
                if len(self._preview_cache) > 20:
                    # Удаляем самый старый элемент
                    oldest_key = next(iter(self._preview_cache))
                    del self._preview_cache[oldest_key]
                
                self.result_image_view.setPixmap(pix)
                self.result_image_view.setToolTip(img_path)
                self.current_result_index = row
            else:
                # Fallback к cv2 если QPixmap не смог загрузить
                img = safe_cv2_imread(img_path)
                if img is not None:
                    pix = QPixmap.fromImage(cv2_to_qimage(img))
                    pix.setDevicePixelRatio(1.0)
                    # Кэшируем результат
                    if not hasattr(self, '_preview_cache'):
                        self._preview_cache = {}
                    self._preview_cache[img_path] = pix
                    # Ограничиваем размер кэша
                    if len(self._preview_cache) > 20:
                        oldest_key = next(iter(self._preview_cache))
                        del self._preview_cache[oldest_key]
                    
                    self.result_image_view.setPixmap(pix)
                    self.result_image_view.setToolTip(img_path)
                    self.current_result_index = row
                else:
                    self.result_image_view.setPixmap(QPixmap())
        except Exception as e:
            logging.error(f"Ошибка загрузки превью: {e}")
            self.result_image_view.setPixmap(QPixmap())

    def get_result_files(self):
        if not self.output_dir or not os.path.isdir(self.output_dir):
            return []
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files = [
            str(Path(self.output_dir) / f)
            for f in sorted(os.listdir(self.output_dir))
            if Path(f).suffix.lower() in exts
        ]
        return files

    def navigate_result(self, delta):
        row = self.result_table.currentRow()
        n = self.result_table.rowCount()
        if n == 0:
            return
        if row < 0:
            row = 0
        else:
            row = (row + delta) % n
        self.result_table.selectRow(row)
        self.show_result_image_from_selection()

    def keyPressEvent(self, e):
        if self.tabs.currentWidget() == self.slider_widget:
            if not self.overlay_chk.isChecked():
                if e.key() == Qt.Key.Key_Right:
                    self.navigate_result(1)
                    return
                elif e.key() == Qt.Key.Key_Left:
                    self.navigate_result(-1)
                    return
            else:
                if e.key() == Qt.Key.Key_Right:
                    self.navigate_tables(1)
                elif e.key() == Qt.Key.Key_Left:
                    self.navigate_tables(-1)
        super().keyPressEvent(e)

    def navigate_tables(self, delta):
        # Перемещаем выделение в обеих таблицах
        curr_a = self.grp_a.table.currentRow()
        new_a = curr_a + delta
        if 0 <= new_a < self.grp_a.table.rowCount():
            self.grp_a.table.selectRow(new_a)

        curr_b = self.grp_b.table.currentRow()
        new_b = curr_b + delta
        if 0 <= new_b < self.grp_b.table.rowCount():
            self.grp_b.table.selectRow(new_b)

    def closeEvent(self, event):
        self.save_state()
        super().closeEvent(event)

    def save_state(self):
        self.grp_a.save_state(self.settings)
        self.grp_b.save_state(self.settings)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("fuzz", self.fuzz_spin.value())
        self.settings.setValue("thick", self.thick_spin.value())
        self.settings.setValue("color", self.color.name())
        self.settings.setValue("radio_all", self.radio_all.isChecked())
        self.settings.setValue("output_dir", self.output_dir)
        self.settings.setValue("dir_a", self.dir_a)
        self.settings.setValue("dir_b", self.dir_b)
        self.settings.setValue("noise", self.noise_chk.isChecked())
        self.settings.setValue("min_area", self.min_area_spin.value())
        self.settings.setValue("gamma", self.gamma_spin.value())
        self.settings.setValue("add_color", self.add_color.name())
        self.settings.setValue("debug", self.debug_chk.isChecked())
        self.settings.setValue("ssim", self.ssim_chk.isChecked())
        self.settings.setValue("match_tolerance", self.match_tolerance_spin.value())
        self.settings.setValue("match_color", self.match_color.name())
        # Сохраняем размеры сплиттеров
        self.settings.setValue("splitter_sizes", self.splitter.sizes())
        self.settings.setValue("main_splitter_sizes", self.main_splitter.sizes())

    def restore_state(self):
        self.grp_a.restore_state(self.settings)
        self.grp_b.restore_state(self.settings)
        geo = self.settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        winstate = self.settings.value("windowState")
        if winstate:
            self.restoreState(winstate)
        fuzz = self.settings.value("fuzz")
        if fuzz:
            self.fuzz_spin.setValue(int(fuzz))
        thick = self.settings.value("thick")
        if thick:
            self.thick_spin.setValue(int(thick))
        color = self.settings.value("color")
        if color:
            self.color = QColor(color)
            self.color_btn.setText(f"Цвет: {self.color.name()}")
            self.color_btn.setStyleSheet(f"background:{self.color.name()}")
        radio_all = self.settings.value("radio_all")
        if radio_all is not None:
            self.radio_all.setChecked(radio_all == "true" or radio_all is True)
        self.output_dir = self.settings.value("output_dir", "")
        self.dir_a = self.settings.value("dir_a", "")
        self.dir_b = self.settings.value("dir_b", "")
        self.grp_a.dir_path = self.dir_a
        self.grp_b.dir_path = self.dir_b
        self.update_out_dir_label()
        self.load_results_from_output_dir()
        noise = self.settings.value("noise")
        if noise is not None:
            self.noise_chk.setChecked(noise == "true" or noise is True)
        min_area = self.settings.value("min_area")
        if min_area:
            self.min_area_spin.setValue(int(min_area))
        gamma = self.settings.value("gamma")
        if gamma:
            self.gamma_spin.setValue(float(gamma))
        add_color = self.settings.value("add_color")
        if add_color:
            self.add_color = QColor(add_color)
            self.add_color_btn.setText(f"Цвет добавленного: {self.add_color.name()}")
            self.add_color_btn.setStyleSheet(f"background:{self.add_color.name()}")
        debug = self.settings.value("debug")
        if debug is not None:
            self.debug_chk.setChecked(debug == "true" or debug is True)
        ssim = self.settings.value("ssim")
        if ssim is not None:
            self.ssim_chk.setChecked(ssim == "true" or ssim is True)
        match_tolerance = self.settings.value("match_tolerance")
        if match_tolerance:
            self.match_tolerance_spin.setValue(int(match_tolerance))
        match_color = self.settings.value("match_color")
        if match_color:
            self.match_color = QColor(match_color)
            self.match_color_btn.setText(f"Цвет совпадений: {self.match_color.name()}")
            self.match_color_btn.setStyleSheet(f"background:{self.match_color.name()}; color:white")
        # Восстанавливаем размеры сплиттеров
        splitter_sizes = self.settings.value("splitter_sizes")
        if splitter_sizes:
            try:
                self.splitter.setSizes([int(x) for x in splitter_sizes])
            except Exception:
                pass
        main_splitter_sizes = self.settings.value("main_splitter_sizes")
        if main_splitter_sizes:
            try:
                self.main_splitter.setSizes([int(x) for x in main_splitter_sizes])
            except Exception:
                pass
        
        # Инициализируем менеджер смещения если есть папка вывода
        if self.output_dir:
            self.alignment_manager = ImageAlignmentManager(self.output_dir)
            if self.alignment_control_panel:
                self.alignment_control_panel.alignment_manager = self.alignment_manager
        
        # Обновляем состояние кнопки сохранения после восстановления состояния
        self.update_save_button_state()

    def update_result_table(self):
        # Показываем все файлы из папки B
        files = self.grp_b.all_files()
        self.result_table.setRowCount(0)
        for f in files:
            name = os.path.basename(f)
            self.result_table.insertRow(self.result_table.rowCount())
            self.result_table.setItem(self.result_table.rowCount()-1, 0, QTableWidgetItem(name))
            self.result_table.setItem(self.result_table.rowCount()-1, 1, QTableWidgetItem(""))
            self.result_table.setItem(self.result_table.rowCount()-1, 2, QTableWidgetItem(f))

    def on_result_selection_changed(self):
        if not self.overlay_chk.isChecked():
            self.show_result_image_from_selection()

    def open_result_external(self):
        row = self.result_table.currentRow()
        if row < 0:
            return
        cell = self.result_table.item(row, 2)
        if cell is not None:
            path = cell.text()
            if os.path.isfile(path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_result_internal_viewer(self):
        row = self.result_table.currentRow()
        if row < 0:
            return
        cell = self.result_table.item(row, 2)
        if cell is not None:
            path = cell.text()
            if os.path.isfile(path):
                img = safe_cv2_imread(path)
                if img is not None:
                    pix = QPixmap.fromImage(cv2_to_qimage(img))
                    pix.setDevicePixelRatio(1.0)
                    self._external_viewer = ExternalResultViewer(pix, title=os.path.basename(path))
                    self._external_viewer.show()

    def save_overlay(self):
        """Сохраняет overlay для выбранных или всех файлов в зависимости от настроек"""
        if not self.overlay_chk.isChecked():
            QMessageBox.warning(self, "Режим overlay не включен", "Сначала включите режим overlay в слайдере.")
            return

        # Получаем файлы в зависимости от выбранного режима
        # При "Сохранить все" (radio_all) игнорируется пользовательский выбор и обрабатываются все файлы
        # При "Сохранить выделенные" (radio_sel) обрабатываются только выбранные файлы
        files_a = self.grp_a.selected_files() if self.radio_sel.isChecked() else self.grp_a.all_files()
        files_b = self.grp_b.selected_files() if self.radio_sel.isChecked() else self.grp_b.all_files()
        
        # Логируем режим работы для отладки
        mode = "выделенные" if self.radio_sel.isChecked() else "все"
        logging.info(f"Режим сохранения: {mode}. Файлов A: {len(files_a)}, файлов B: {len(files_b)}")
        
        # Проверяем количество файлов
        if not files_a or not files_b:
            QMessageBox.warning(self, "Нет файлов для обработки", "Выберите файлы в обеих папках.")
            return
            
        if self.radio_sel.isChecked() and len(files_a) != len(files_b):
            QMessageBox.warning(self, "Несимметричный выбор", 
                              f"Выделено {len(files_a)} файлов в папке A и {len(files_b)} файлов в папке B.\n"
                              "Выделите одинаковое количество файлов в обеих папках.")
            return
            
        if not self.output_dir:
            QMessageBox.warning(self, "Нет папки вывода", "Сначала выберите папку для результатов.")
            return

        # Импортируем gc для управления памятью
        import gc
        
        self.progress_bar.setMaximum(len(files_a))
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # Получаем все файлы из папки результатов
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        all_result_files = [
            str(Path(self.output_dir) / f)
            for f in sorted(os.listdir(self.output_dir))
            if Path(f).suffix.lower() in exts
        ]
        self.result_table.setRowCount(0)
        
        # Заполняем грид всеми файлами из папки результатов
        for f in all_result_files:
            name = os.path.basename(f)
            self.result_table.insertRow(self.result_table.rowCount())
            self.result_table.setItem(self.result_table.rowCount()-1, 0, QTableWidgetItem(name))
            self.result_table.setItem(self.result_table.rowCount()-1, 1, QTableWidgetItem(""))
            self.result_table.setItem(self.result_table.rowCount()-1, 2, QTableWidgetItem(f))
        
        # Словарь для быстрого поиска строки по имени
        name_to_row = {self.result_table.item(row, 0).text(): row for row in range(self.result_table.rowCount())}
        
        success_count = 0
        error_count = 0
        
        for i, (a, b) in enumerate(zip(files_a, files_b)):
            # Обрабатываем события Qt для отзывчивости UI
            QApplication.processEvents()
            
            out_name = f"{Path(a).stem}__overlay__{Path(b).stem}.png"
            out_path = Path(self.output_dir) / out_name
            
            # Обновляем прогресс с информацией о текущем файле
            self.progress_bar.setFormat(f"Обработка overlay: {Path(a).name} vs {Path(b).name}")
            
            try:
                # Загружаем исходные изображения в полном разрешении
                img_a_cv = safe_cv2_imread(a)
                img_b_cv = safe_cv2_imread(b)
                
                if img_a_cv is None or img_b_cv is None:
                    raise Exception(f"Не удалось загрузить одно из изображений: {a} или {b}")

                # Конвертируем в QImage для обработки
                img_a = QPixmap.fromImage(cv2_to_qimage(img_a_cv))
                img_b = QPixmap.fromImage(cv2_to_qimage(img_b_cv))
                
                # Создаем временный SliderReveal для генерации overlay в полном разрешении
                temp_slider = SliderReveal(img_a, img_b)
                temp_slider.setOverlayMode(True)
                
                # Генерируем overlay в полном разрешении
                overlay_qimage = temp_slider._generate_overlay_cache()
                
                if overlay_qimage is None:
                    raise Exception("Не удалось сгенерировать overlay")

                # Сохраняем в файл
                success = overlay_qimage.save(str(out_path), "PNG")
                
                if not success:
                    raise Exception(f"Не удалось сохранить файл: {out_path}")
                
                # Добавляем результат в таблицу
                row = name_to_row.get(out_name)
                if row is not None:
                    self.result_table.setItem(row, 1, QTableWidgetItem("OK"))
                else:
                    # Если файла еще нет в таблице, добавляем его
                    self.result_table.insertRow(self.result_table.rowCount())
                    self.result_table.setItem(self.result_table.rowCount()-1, 0, QTableWidgetItem(out_name))
                    self.result_table.setItem(self.result_table.rowCount()-1, 1, QTableWidgetItem("OK"))
                    self.result_table.setItem(self.result_table.rowCount()-1, 2, QTableWidgetItem(str(out_path)))
                
                success_count += 1
                
            except Exception as e:
                logging.error(f"Ошибка сохранения overlay для {a} vs {b}: {e}")
                status = f"Error: {e}"
                row = name_to_row.get(out_name)
                if row is not None:
                    self.result_table.setItem(row, 1, QTableWidgetItem(status))
                error_count += 1
            
            self.progress_bar.setValue(i + 1)
            
            # Принудительная сборка мусора каждые 5 файлов для больших изображений
            if (i + 1) % 5 == 0:
                gc.collect()
                QApplication.processEvents()
        
        self.progress_bar.hide()
        self.progress_bar.setFormat("")  # Сбрасываем формат
        
        # Финальная сборка мусора
        gc.collect()
        
        # Уведомление о завершении сохранения с детальной статистикой
        mode = "выделенные" if self.radio_sel.isChecked() else "все"
        message = f"Сохранение overlay завершено!\n\n"
        message += f"Режим: {mode}\n"
        message += f"Обработано пар изображений: {len(files_a)}\n"
        message += f"Успешно: {success_count}\n"
        message += f"Ошибок: {error_count}\n\n"
        message += f"Результаты сохранены в папку:\n{self.output_dir}"
        
        if error_count > 0:
            QMessageBox.warning(self, "Сохранение завершено с ошибками", message)
        else:
            QMessageBox.information(self, "Сохранение завершено", message)

    def update_save_button_state(self):
        """Обновляет состояние кнопки сохранения на основе текущих настроек"""
        # Включаем кнопку сохранения только если overlay включен И есть файлы для обработки
        files_a = self.grp_a.selected_files() if self.radio_sel.isChecked() else self.grp_a.all_files()
        files_b = self.grp_b.selected_files() if self.radio_sel.isChecked() else self.grp_b.all_files()
        has_files = len(files_a) > 0 and len(files_b) > 0 and len(files_a) == len(files_b)
        has_output_dir = bool(self.output_dir)
        overlay_checked = self.overlay_chk.isChecked()
        
        # Упрощенная логика: включаем кнопку если overlay включен
        # Остальные проверки будут выполнены в save_overlay методе
        should_enable = overlay_checked
        self.save_overlay_btn.setEnabled(should_enable)
    
    def on_alignment_changed(self, offset_x: int, offset_y: int):
        """Обработчик изменения смещения изображений"""
        # Обновляем слайдер с новым смещением
        self.update_slider_with_alignment()
    
    def update_slider_with_alignment(self):
        """Обновляет слайдер с учетом смещения изображений"""
        files_a = self.grp_a.selected_files()
        files_b = self.grp_b.selected_files()
        file_a = files_a[0] if files_a else None
        file_b = files_b[0] if files_b else None

        self.label_a.setText(f"A: {Path(file_a).name if file_a else '<не выбрано>'}")
        self.label_b.setText(f"B: {Path(file_b).name if file_b else '<не выбрано>'}")

        if file_a and file_b and self.alignment_manager:
            # Загружаем изображения в оригинальном разрешении
            img_a_cv = safe_cv2_imread(file_a)
            img_b_cv = safe_cv2_imread(file_b)
            if img_a_cv is not None and img_b_cv is not None:
                img_a = QPixmap.fromImage(cv2_to_qimage(img_a_cv))
                img_b = QPixmap.fromImage(cv2_to_qimage(img_b_cv))
                img_a.setDevicePixelRatio(1.0)
                img_b.setDevicePixelRatio(1.0)
                
                # Применяем смещение
                aligned_a, aligned_b = self.alignment_manager.apply_alignment_to_pixmaps(
                    img_a, img_b, file_a, file_b
                )
                
                self.slider_reveal.setPixmaps(aligned_a, aligned_b)
                self.slider_reveal.setVisible(True)
                
                # Обновляем панель управления смещением
                if self.alignment_control_panel:
                    self.alignment_control_panel.set_current_images(file_a, file_b, img_a, img_b)
                
                # Обновляем состояние кнопки сохранения
                self.update_save_button_state()
            else:
                self.slider_reveal.setVisible(False)
                self.save_overlay_btn.setEnabled(False)
        else:
            self.slider_reveal.setVisible(False)
            self.save_overlay_btn.setEnabled(False)


class ExternalResultViewer(QWidget):
    def __init__(self, pixmap, title="Просмотр результата"):
        super().__init__()
        self.setWindowTitle(title)
        self.viewer = ResultImageView()
        layout = QVBoxLayout(self)
        layout.addWidget(self.viewer)
        self.viewer.setPixmap(pixmap)
        self.resize(1200, 800)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Кастомизация Fusion
    palette = app.palette()
    palette.setColor(palette.Window, QColor(245, 245, 245))
    palette.setColor(palette.Base, QColor(255, 255, 255))
    palette.setColor(palette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(palette.Text, QColor(30, 30, 30))
    palette.setColor(palette.Button, QColor(230, 230, 230))
    palette.setColor(palette.ButtonText, QColor(30, 30, 30))
    palette.setColor(palette.Highlight, QColor(0, 120, 215))
    palette.setColor(palette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


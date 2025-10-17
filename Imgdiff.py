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

# ВРЕМЕННО ОТКЛЮЧЕНО: Функциональность смещения изображений
# Панель управления смещением скрыта, будет реализована позже
# ✅ НАВИГАЦИЯ РАБОТАЕТ: Кнопки ◀▶ для переключения между парами изображений

# flake8: noqa: E402
from PyQt5.QtCore import Qt, QUrl, QSettings, pyqtSignal, QPoint, QTimer, QPropertyAnimation, QEasingCurve, QMimeData, QRect, QObject, QRunnable, QThreadPool
from PyQt5.QtGui import QPixmap, QDesktopServices, QColor, QImage, QPainter, QKeySequence, QDrag, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel, QFileDialog,
    QGroupBox, QRadioButton, QMessageBox, QSplitter, QSpinBox, QFormLayout,
    QButtonGroup, QColorDialog, QDoubleSpinBox, QTabWidget,
    QComboBox, QProgressBar, QSizePolicy, QCheckBox, QShortcut, QMenu
)
# Дублирующиеся импорты убраны

from core.diff_two_color import diff_two_color
from core.slider_reveal import SliderReveal
# Временно отключена функциональность смещения изображенийcl
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

MAX_PREVIEW_SIZE = 1200

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
    img = fast_cv2_imread(path)
    if img is None: return QPixmap()
    h, w = img.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return QPixmap.fromImage(cv2_to_qimage(img))

def quick_diff_ratio(img_a: np.ndarray, img_b: np.ndarray, max_side: int = 256, thr: int = 5) -> float:
    """Быстрый оценочный процент отличий на даунскейле.
    Возвращает долю пикселей (0..1) где |A-B| > thr в градациях серого.
    """
    try:
        h, w = img_a.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            a = cv2.resize(img_a, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            b = cv2.resize(img_b, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            a, b = img_a, img_b
        ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ag, bg)
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        return float(cv2.countNonZero(mask)) / float(mask.size)
    except Exception:
        return 1.0

def fast_cv2_imread(path):
    """Быстрое чтение изображения: cv2.imread -> fallback на imdecode."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        with open(path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

class WorkerSignals(QObject):
    finished = pyqtSignal(str, str, int, str)  # out_name, out_path, code, error_message


class CompareWorker(QRunnable):
    def __init__(self, a, b, out_path, params):
        super().__init__()
        self.a = a
        self.b = b
        self.out_path = out_path
        self.params = params
        self.signals = WorkerSignals()

    def run(self):
        try:
            code = run_outline_core(
                self.a,
                self.b,
                self.out_path,
                self.params['fuzz'],
                self.params['thick'],
                self.params['del_color_bgr'],
                self.params['add_color_bgr'],
                self.params['match_tolerance'],
                self.params['match_color_bgr'],
                self.params['gamma'],
                self.params['morph_open'],
                self.params['min_area'],
                self.params['debug'],
                self.params['use_ssim'],
                self.params['output_dir'],
            )
            self.signals.finished.emit(self.params['out_name'], str(self.out_path), code, "")
        except Exception as e:
            self.signals.finished.emit(self.params['out_name'], str(self.out_path), -1, str(e))

def run_outline_core(left, right, out_path, fuzz, thick, del_color_bgr, add_color_bgr,
                     match_tolerance, match_color_bgr, gamma, morph_open, min_area,
                     debug, use_ssim, output_dir):
    """Потокобезопасное сравнение пары изображений с сохранением результата.
    Возвращает 1 если есть отличия, 0 если равны.
    """
    old = fast_cv2_imread(str(left))
    new = fast_cv2_imread(str(right))
    if old is None or new is None:
        raise FileNotFoundError(f"Не удалось загрузить {left} или {right}")

    # Выравниваем размеры
    h = max(old.shape[0], new.shape[0])
    w = max(old.shape[1], new.shape[1])
    if old.shape[:2] != (h, w):
        old = cv2.resize(old, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if new.shape[:2] != (h, w):
        new = cv2.resize(new, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Быстрый ранний фильтр на даунскейле
    try:
        ratio = quick_diff_ratio(old, new, max_side=256, thr=5)
        if ratio < 0.001:
            del old, new
            return 0
    except Exception:
        pass

    debug_dir = Path(output_dir) / 'debug' if debug and output_dir else Path('.')

    overlay, meta = diff_two_color(
        old_img=old,
        new_img=new,
        sens=fuzz,
        blur=3,
        morph_open=morph_open,
        min_area=min_area,
        kernel=thick,
        alpha=0.6,
        gamma=gamma,
        del_color=del_color_bgr,
        add_color=add_color_bgr,
        debug=debug,
        debug_dir=debug_dir,
        use_ssim=use_ssim,
        match_tolerance=match_tolerance,
        match_color=match_color_bgr
    )

    if meta.get('diff_pixels', 0) > 0:
        cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    del old, new, overlay
    return 1 if meta.get('diff_pixels', 0) > 0 else 0

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
    
    def startDrag(self, actions):
        """Начать перетаскивание выбранных элементов"""
        try:
            selected_items = self.selectedItems()
            if not selected_items:
                return
            
            # Создаем MIME данные с путями к файлам
            mime_data = QMimeData()
            urls = []
            file_paths = []
            
            for item in selected_items:
                if item.column() == 0:  # Только элементы первой колонки (имя файла)
                    file_path = item.data(Qt.UserRole)
                    if file_path and os.path.isfile(file_path):
                        urls.append(QUrl.fromLocalFile(file_path))
                        file_paths.append(file_path)
            
            if urls:
                mime_data.setUrls(urls)
                mime_data.setText('\n'.join(file_paths))
                
                # Создаем и начинаем перетаскивание
                drag = QDrag(self)
                drag.setMimeData(mime_data)
                
                # Создаем пиктограмму для перетаскивания
                pixmap = QPixmap(100, 30)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setPen(QPen(Qt.black))
                painter.drawText(pixmap.rect(), Qt.AlignCenter, f"{len(file_paths)} файл(ов)")
                painter.end()
                
                drag.setPixmap(pixmap)
                drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
                
                # Выполняем перетаскивание
                drag.exec(Qt.CopyAction)
                
        except Exception as e:
            logging.error(f"Ошибка при начале перетаскивания: {e}")

class FilteredTable(QWidget):
    def __init__(self, label, settings_key, parent=None):
        super().__init__(parent)
        self.settings_key = settings_key
        self._layout = QVBoxLayout(self)
        
        # 📍 Заголовок с меткой выбранной папки
        header_layout = QHBoxLayout()
        self.dir_btn = QPushButton(label)
        self.dir_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        
        # 🏷️ Метка для отображения выбранного пути
        self.path_label = QLabel("Папка не выбрана")
        self.path_label.setStyleSheet("color: #666; font-style: italic; padding: 4px;")
        self.path_label.setWordWrap(True)
        self.path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        header_layout.addWidget(self.dir_btn)
        header_layout.addWidget(self.path_label)
        
        # --- 🔍 Новый фильтр: QComboBox с историей ---
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
        
        # 🎯 Кнопки с иконками и подсказками
        # ✕ Кнопка сброса фильтра
        self.clear_filter_btn = QPushButton('✕')
        self.clear_filter_btn.setToolTip("Сбросить фильтр")
        self.clear_filter_btn.setFixedWidth(32)
        self.clear_filter_btn.setStyleSheet("""
            QPushButton {
                background: #ff6b6b;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff5252;
            }
        """)
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        
        # 🗑 Кнопка очистки истории
        self.clear_history_btn = QPushButton('🗑')
        self.clear_history_btn.setToolTip("Очистить историю фильтров")
        self.clear_history_btn.setFixedWidth(32)
        self.clear_history_btn.setStyleSheet("""
            QPushButton {
                background: #ffa726;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #ff9800;
            }
        """)
        self.clear_history_btn.clicked.connect(self.clear_filter_history)
        
        # 🔄 Кнопка обновления директории
        self.refresh_btn = QPushButton('🔄')
        self.refresh_btn.setToolTip("Обновить содержимое папки")
        self.refresh_btn.setFixedWidth(32)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background: #42a5f5;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #2196f3;
            }
        """)
        self.refresh_btn.clicked.connect(self.refresh_dir)
        
        # ↑↓ Кнопки сортировки
        self.sort_asc_btn = QPushButton('↑')
        self.sort_asc_btn.setToolTip("Сортировка по возрастанию")
        self.sort_asc_btn.setFixedWidth(32)
        self.sort_asc_btn.setStyleSheet("""
            QPushButton {
                background: #66bb6a;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #4caf50;
            }
        """)
        self.sort_asc_btn.clicked.connect(self.sort_ascending)
        
        self.sort_desc_btn = QPushButton('↓')
        self.sort_desc_btn.setToolTip("Сортировка по убыванию")
        self.sort_desc_btn.setFixedWidth(32)
        self.sort_desc_btn.setStyleSheet("""
            QPushButton {
                background: #66bb6a;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #4caf50;
            }
        """)
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
        
        # Включаем поддержку перетаскивания элементов таблицы
        self.table.setDragEnabled(True)
        self.table.setDragDropMode(QTableWidget.DragDropMode.DragOnly)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(200)
        
        self._layout.addLayout(header_layout)
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

    def update_path_label(self):
        """🏷️ Обновляет метку с путем выбранной папки"""
        if self.dir_path:
            # 📂 Показываем только имя папки и родительскую директорию для краткости
            path_obj = Path(self.dir_path)
            if path_obj.parent.name:
                display_path = f"{path_obj.parent.name} / {path_obj.name}"
            else:
                display_path = path_obj.name
            self.path_label.setText(display_path)
            self.path_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 4px; background: #e8f5e8; border-radius: 4px;")
        else:
            self.path_label.setText("Папка не выбрана")
            self.path_label.setStyleSheet("color: #666; font-style: italic; padding: 4px;")

    def load_files(self, files, dir_path=None):
        # 🔄 Сортируем файлы с естественной сортировкой при загрузке
        sorted_files = sorted(files, key=lambda f: natural_sort_key(os.path.basename(f)))
        self.files = [(os.path.basename(f), f) for f in sorted_files]
        if dir_path:
            self.dir_path = dir_path
        self.update_path_label()  # 🏷️ Обновляем метку пути
        self.apply_filter()  # Применяем фильтр, который добавит файлы в таблицу с Qt.UserRole

    def load_from_dir(self, dir_path):
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files = [
            str(Path(dir_path) / f)
            for f in os.listdir(dir_path)
            if Path(f).suffix.lower() in exts
        ]
        self.load_files(files, dir_path)

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

    def apply_filter(self):
        text = self.filter_combo.currentText().lower()
        self.table.setRowCount(0)
        self.filtered = []
        
        # 🔍 Фильтруем файлы
        filtered_files = []
        for idx, (name, path) in enumerate(self.files):
            if text in name.lower():
                filtered_files.append((idx, name, path))
        
        # 🔄 Сортируем отфильтрованные файлы с естественной сортировкой
        if self.sort_order == "asc":
            filtered_files.sort(key=lambda x: natural_sort_key(x[1]))  # 📈 по имени по возрастанию
        elif self.sort_order == "desc":
            filtered_files.sort(key=lambda x: natural_sort_key(x[1]), reverse=True)  # 📉 по имени по убыванию
        
        # ➕ Добавляем в таблицу
        for idx, name, path in filtered_files:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # Создаем элемент с именем файла
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, path)  # Сохраняем полный путь
            name_item.setToolTip(path)  # Показываем полный путь при наведении
            
            self.table.setItem(row, 0, name_item)
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
    
    def invalidate_overlay_cache(self):
        """Инвалидирует кэш overlay при изменении цветов"""
        self._overlay_cache = None
        self._overlay_cache_key = None
        self.update()

    def _generate_overlay_cache(self):
        """Генерирует кэшированное overlay изображение с цветами из настроек"""
        if self.pixmap_a is None or self.pixmap_b is None:
            return None
            
        # Получаем ссылку на главное окно для доступа к цветам
        main_window = self.parent()
        while main_window and not hasattr(main_window, 'color'):
            main_window = main_window.parent()
        
        if not main_window or not hasattr(main_window, 'color'):
            # Fallback: используем цвета по умолчанию
            color_a = QColor("#FF0000")  # Красный
            color_b = QColor("#0066FF")  # Синий
            color_match = QColor("#0000FF")  # Синий
        else:
            color_a = main_window.color
            color_b = main_window.add_color
            color_match = main_window.match_color
            
        # Создаем ключ кэша на основе ID изображений и цветов
        cache_key = (id(self.pixmap_a), id(self.pixmap_b), 
                    color_a.name(), color_b.name(), color_match.name())
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
            
            # ИСПРАВЛЕНИЕ: Используем цвета из настроек вместо жестко заданных
            # Получаем цвета из настроек и конвертируем в RGB
            color_a_rgb = [color_a.red(), color_a.green(), color_a.blue()]
            color_b_rgb = [color_b.red(), color_b.green(), color_b.blue()]
            color_match_rgb = [color_match.red(), color_match.green(), color_match.blue()]
            
            # ОПТИМИЗАЦИЯ: Векторизованное применение цветов
            # Применяем цвета пакетно для всех пикселей одновременно
            
            # Цвет из настроек для изображения A (полупрозрачный)
            out[mask_a] = color_a_rgb + [120]  # RGBA: цвет A с alpha=120
            
            # Цвет добавленного только для уникальных пикселей B (не пересекающихся с A)
            only_b = mask_b & ~mask_a  # Логическое И: B И НЕ A
            out[only_b] = color_b_rgb + [180]  # RGBA: цвет добавленного с alpha=180
            
            # Цвет совпадений для пересекающихся областей (A И B)
            both = mask_a & mask_b  # Логическое И: A И B
            out[both] = color_match_rgb + [200]  # RGBA: цвет совпадений с alpha=200
            
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
            # Рисуем частями без копирования буферов
            qp.drawPixmap(QRect(0, 0, split_x, self.pixmap_a.height()), self.pixmap_a,
                          QRect(0, 0, split_x, self.pixmap_a.height()))
            qp.drawPixmap(QRect(split_x, 0, self.pixmap_b.width() - split_x, self.pixmap_b.height()),
                          self.pixmap_b,
                          QRect(split_x, 0, self.pixmap_b.width() - split_x, self.pixmap_b.height()))
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
        
        # Инициализируем менеджер смещения изображений (временно отключен)
        self.alignment_manager = None  # Будет инициализирован при выборе папки вывода
        self.alignment_control_panel = None  # Будет создан при инициализации UI
        
        print('step 2')
        # Пул потоков для параллельной обработки
        self.threadpool = QThreadPool.globalInstance()
        try:
            import os as _os
            self.threadpool.setMaxThreadCount(max(1, min((_os.cpu_count() or 4), 8)))
        except Exception:
            pass
        self.batch_total = 0
        self.batch_done = 0
        self.batch_ok = 0
        self.batch_equal = 0
        self.batch_err = 0
        # --- 🔘 Радиокнопки сравнения в QGroupBox ---
        self.radio_all = QRadioButton("Сравнить все")
        self.radio_sel = QRadioButton("Сравнить только выделенные")
        self.radio_sel.setChecked(True)
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_all)
        self.radio_group.addButton(self.radio_sel)
        radio_box = QGroupBox("⚙️ Режим сравнения")
        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.radio_all)
        radio_layout.addWidget(self.radio_sel)
        radio_box.setLayout(radio_layout)
        radio_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #424242;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        print('step 3')
        self.compare_btn = QPushButton("⚡ Сравнить")
        self.compare_btn.setStyleSheet("""
            QPushButton {
                background: #ff9800;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #f57c00;
            }
        """)
        self.compare_btn.clicked.connect(self.compare_parallel)
        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(["Имя", "Статус", ""])
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setColumnHidden(2, True)
        self.result_table.itemDoubleClicked.connect(self.open_result)
        print('step 4')
        self.out_dir_label = QLabel("📤 Папка вывода:")
        self.out_dir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.out_dir_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #d84315;
                padding: 8px;
                background: #fbe9e7;
                border-radius: 6px;
                margin: 4px;
            }
        """)
        self.out_dir_btn = QPushButton("📁 Выбрать папку вывода…")
        self.out_dir_btn.setStyleSheet("""
            QPushButton {
                background: #ff7043;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff5722;
            }
        """)
        self.out_dir_btn.clicked.connect(self.choose_out_dir)
        self.out_dir_refresh_btn = QPushButton('🔄')
        self.out_dir_refresh_btn.setToolTip("Обновить список результатов")
        self.out_dir_refresh_btn.setFixedWidth(32)
        self.out_dir_refresh_btn.setStyleSheet("""
            QPushButton {
                background: #42a5f5;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #2196f3;
            }
        """)
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
        results_label = QLabel("📊 Результаты:")
        results_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #6a1b9a;
                padding: 8px;
                background: #f3e5f5;
                border-radius: 6px;
                margin: 4px;
            }
        """)
        result_col.addWidget(results_label)
        result_col_w = QWidget()
        result_col_w.setLayout(result_col)
        result_col_w.setMinimumWidth(120)
        result_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        print('step 6')
        # --- Вкладка 1: ⚙️ Настройки сравнения ---
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
        param_group = QGroupBox("🔧 Параметры Outline")
        param_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #424242;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        param_group.setLayout(param_form)
        param_group.setMaximumWidth(350)
        # --- 📚 Пояснения отдельным блоком ---
        param_help = QLabel(
            "<b>📚 Пояснения к параметрам:</b><br>"
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

        # --- Вкладка 2: 🔄 Сравнение/Слайдер ---
        # 📁 Левый столбец: грид A
        print('left_col_w start')
        self.grp_a = FilteredTable("📁 Папка A", "A")
        self.grp_a_label = QLabel("📁 Папка A")
        self.grp_a_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grp_a_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #1976d2;
                padding: 8px;
                background: #e3f2fd;
                border-radius: 6px;
                margin: 4px;
            }
        """)
        left_col = QVBoxLayout()
        left_col.addWidget(self.grp_a_label)
        left_col.addWidget(self.grp_a)
        
        # 🚫 Таблица исключений для папки A
        self.exclude_a_label = QLabel("🚫 Исключения A")
        self.exclude_a_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exclude_a_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #d32f2f;
                padding: 4px;
                background: #ffebee;
                border-radius: 4px;
                margin: 2px;
            }
        """)
        self.exclude_a_table = QTableWidget()
        self.exclude_a_table.setColumnCount(1)
        self.exclude_a_table.setHorizontalHeaderLabels(["Исключенные файлы"])
        self.exclude_a_table.setMaximumHeight(120)
        self.exclude_a_table.setDragDropMode(QTableWidget.DragDropMode.DropOnly)
        self.exclude_a_table.setAcceptDrops(True)
        self.exclude_a_table.dropEvent = self.exclude_a_drop_event
        self.exclude_a_table.dragEnterEvent = self.exclude_a_drag_enter_event
        self.exclude_a_table.dragMoveEvent = self.exclude_a_drag_move_event
        
        left_col.addWidget(self.exclude_a_label)
        left_col.addWidget(self.exclude_a_table)
        
        # 🔄 Кнопка возврата файлов из исключений A
        self.restore_a_btn = QPushButton("↩️ Вернуть файлы")
        self.restore_a_btn.setToolTip("Вернуть выбранные файлы из исключений")
        self.restore_a_btn.setStyleSheet("""
            QPushButton {
                background: #4caf50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #388e3c;
            }
            QPushButton:disabled {
                background: #bdbdbd;
                color: #757575;
            }
        """)
        self.restore_a_btn.clicked.connect(self.restore_excluded_files_a)
        self.restore_a_btn.setEnabled(False)
        
        left_col.addWidget(self.restore_a_btn)
        
        left_col_w = QWidget()
        left_col_w.setLayout(left_col)
        left_col_w.setMinimumWidth(120)
        left_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        print('left_col_w end')
        print('step 7')
        # 📁 Средний столбец: грид B
        print('mid_col_w start')
        self.grp_b = FilteredTable("📁 Папка B", "B")
        self.grp_b_label = QLabel("📁 Папка B")
        self.grp_b_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grp_b_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #388e3c;
                padding: 8px;
                background: #e8f5e8;
                border-radius: 6px;
                margin: 4px;
            }
        """)
        mid_col = QVBoxLayout()
        mid_col.addWidget(self.grp_b_label)
        mid_col.addWidget(self.grp_b)
        
        # 🚫 Таблица исключений для папки B
        self.exclude_b_label = QLabel("🚫 Исключения B")
        self.exclude_b_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exclude_b_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #d32f2f;
                padding: 4px;
                background: #ffebee;
                border-radius: 4px;
                margin: 2px;
            }
        """)
        self.exclude_b_table = QTableWidget()
        self.exclude_b_table.setColumnCount(1)
        self.exclude_b_table.setHorizontalHeaderLabels(["Исключенные файлы"])
        self.exclude_b_table.setMaximumHeight(120)
        self.exclude_b_table.setDragDropMode(QTableWidget.DragDropMode.DropOnly)
        self.exclude_b_table.setAcceptDrops(True)
        self.exclude_b_table.dropEvent = self.exclude_b_drop_event
        self.exclude_b_table.dragEnterEvent = self.exclude_b_drag_enter_event
        self.exclude_b_table.dragMoveEvent = self.exclude_b_drag_move_event
        
        mid_col.addWidget(self.exclude_b_label)
        mid_col.addWidget(self.exclude_b_table)
        
        # 🔄 Кнопка возврата файлов из исключений B
        self.restore_b_btn = QPushButton("↩️ Вернуть файлы")
        self.restore_b_btn.setToolTip("Вернуть выбранные файлы из исключений")
        self.restore_b_btn.setStyleSheet("""
            QPushButton {
                background: #4caf50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #388e3c;
            }
            QPushButton:disabled {
                background: #bdbdbd;
                color: #757575;
            }
        """)
        self.restore_b_btn.clicked.connect(self.restore_excluded_files_b)
        self.restore_b_btn.setEnabled(False)
        
        mid_col.addWidget(self.restore_b_btn)
        
        mid_col_w = QWidget()
        mid_col_w.setLayout(mid_col)
        mid_col_w.setMinimumWidth(120)
        mid_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        print('mid_col_w end')
        print('step 8')
        # 📊 Правая колонка: результаты
        self.out_dir_label = QLabel("📤 Папка вывода:")
        self.out_dir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.out_dir_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #d84315;
                padding: 8px;
                background: #fbe9e7;
                border-radius: 6px;
                margin: 4px;
            }
        """)
        self.out_dir_btn = QPushButton("📁 Выбрать папку вывода…")
        self.out_dir_btn.setStyleSheet("""
            QPushButton {
                background: #ff7043;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff5722;
            }
        """)
        self.out_dir_btn.clicked.connect(self.choose_out_dir)
        # 🔄 Кнопка обновления папки результатов
        self.out_dir_refresh_btn = QPushButton('🔄')
        self.out_dir_refresh_btn.setToolTip("Обновить список результатов")
        self.out_dir_refresh_btn.setFixedWidth(32)
        self.out_dir_refresh_btn.setStyleSheet("""
            QPushButton {
                background: #42a5f5;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #2196f3;
            }
        """)
        self.out_dir_refresh_btn.clicked.connect(self.load_results_from_output_dir)
        out_dir_row = QHBoxLayout()
        out_dir_row.addWidget(self.out_dir_btn)
        out_dir_row.addWidget(self.out_dir_refresh_btn)
        # --- 🔘 Радиокнопки сравнения в QGroupBox ---
        self.radio_all = QRadioButton("Сравнить все")
        self.radio_sel = QRadioButton("Сравнить только выделенные")
        self.radio_sel.setChecked(True)
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_all)
        self.radio_group.addButton(self.radio_sel)
        radio_box = QGroupBox("⚙️ Режим сравнения")
        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.radio_all)
        radio_layout.addWidget(self.radio_sel)
        radio_box.setLayout(radio_layout)
        radio_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #424242;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # 💾 Кнопка сохранения overlay - перемещена под радио-кнопки
        self.save_overlay_btn = QPushButton("💾 Сохранить overlay")
        self.save_overlay_btn.setToolTip("Сохранить overlay для выбранных или всех файлов (в зависимости от выделения)")
        self.save_overlay_btn.setStyleSheet("""
            QPushButton {
                background: #2196f3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1976d2;
            }
            QPushButton:disabled {
                background: #bdbdbd;
                color: #757575;
            }
        """)
        self.save_overlay_btn.clicked.connect(self.save_overlay)
        self.save_overlay_btn.setEnabled(False)  # Включаем только когда overlay активен
        

        
        result_col = QVBoxLayout()
        result_col.addWidget(self.out_dir_label)
        result_col.addLayout(out_dir_row)
        result_col.addWidget(radio_box)
        result_col.addWidget(self.save_overlay_btn)  # Кнопка под радио-кнопками
        results_label = QLabel("📊 Результаты:")
        results_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #6a1b9a;
                padding: 8px;
                background: #f3e5f5;
                border-radius: 6px;
                margin: 4px;
            }
        """)
        result_col.addWidget(results_label)
        result_col.addWidget(self.result_table, 1)
        self.open_external_btn = QPushButton("🖼️ Открыть в системном просмотрщике")
        self.open_external_btn.setToolTip("Открыть выбранный результат в стандартном приложении Windows")
        self.open_external_btn.setStyleSheet("""
            QPushButton {
                background: #607d8b;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #455a64;
            }
        """)
        self.open_external_btn.clicked.connect(self.open_result_external)
        result_col.addWidget(self.open_external_btn)
        self.open_internal_viewer_btn = QPushButton("🔍 Открыть в отдельном окне")
        self.open_internal_viewer_btn.setToolTip("Показать выбранный результат в отдельном окне с zoom/drag")
        self.open_internal_viewer_btn.setStyleSheet("""
            QPushButton {
                background: #795548;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #5d4037;
            }
        """)
        self.open_internal_viewer_btn.clicked.connect(self.open_result_internal_viewer)
        result_col.addWidget(self.open_internal_viewer_btn)
        result_col_w = QWidget()
        result_col_w.setLayout(result_col)
        result_col_w.setMinimumWidth(120)
        result_col_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        print('result_col_w end')
        print('step 9')
        
        # 🎛️ Кнопка скрытия/показа панели выбора папок
        self.toggle_folders_btn = QPushButton("👁️ Скрыть панели")
        self.toggle_folders_btn.setToolTip("Скрыть/показа панели выбора папок для увеличения рабочего пространства (Ctrl+H)")
        self.toggle_folders_btn.setStyleSheet("""
            QPushButton {
                background: #607d8b;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #455a64;
            }
        """)
        self.toggle_folders_btn.clicked.connect(self.toggle_folders_panel)
        
        # Горячая клавиша Ctrl+H для скрытия/показа панелей
        self.toggle_folders_shortcut = QShortcut(QKeySequence("Ctrl+H"), self)
        self.toggle_folders_shortcut.activated.connect(self.toggle_folders_panel)
        
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
        # --- 🖼️ Слайдер справа ---
        print('before slider_widget')
        self.slider_widget = QWidget()
        self.slider_layout = QVBoxLayout(self.slider_widget)
        print('after slider_widget')
        # --- 🎛️ Панель управления над слайсером ---
        self.slider_control = QHBoxLayout()
        self.overlay_chk = QCheckBox("Overlay")
        self.overlay_chk.setChecked(False)
        self.overlay_chk.setToolTip("Включить режим наложения (A=красный, B=зелёный)")
        self.overlay_chk.stateChanged.connect(self.update_slider_overlay_mode)
        
        # Кнопка "Вписать всё"
        self.fit_to_window_btn = QPushButton("🔍 Вписать всё")
        self.fit_to_window_btn.setToolTip("Вписать изображение в окно целиком")
        self.fit_to_window_btn.setStyleSheet("""
            QPushButton {
                background: #4caf50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #388e3c;
            }
        """)
        self.fit_to_window_btn.clicked.connect(self.fit_to_window)
        
        # Кнопка "Подсветить различия"
        self.highlight_diff_btn = QPushButton("💡 Подсветить различия")
        self.highlight_diff_btn.setToolTip("Подсветить места различий мигающим кругом на 3 секунды")
        self.highlight_diff_btn.setStyleSheet("""
            QPushButton {
                background: #ff9800;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #f57c00;
            }
            QPushButton:disabled {
                background: #bdbdbd;
                color: #757575;
            }
        """)
        self.highlight_diff_btn.clicked.connect(self.highlight_differences)
        self.highlight_diff_btn.setEnabled(False)  # Включаем только когда overlay активен
        
        # Кнопка отладки различий (УБРАНА)
        # self.debug_diff_btn = QPushButton("🐛 Debug различия")
        # self.debug_diff_btn.setToolTip("Показать отладочную информацию о различиях")
        # self.debug_diff_btn.setStyleSheet("""
        #     QPushButton {
        #         background: #ff9800;
        #         color: white;
        #         border: none;
        #         border-radius: 6px;
        #         padding: 6px 12px;
        #         font-weight: bold;
        #     }
        #     QPushButton:hover {
        #         background: #f57c00;
        #     }
        # """)
        # self.debug_diff_btn.clicked.connect(self.debug_differences)
        # self.debug_diff_btn.setEnabled(False)  # Включаем только когда overlay активен
        

        
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(32)
        self.prev_btn.setToolTip("Предыдущая пара изображений (←)")
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background: #9c27b0;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #7b1fa2;
            }
        """)
        self.prev_btn.clicked.connect(self.navigate_previous)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(32)
        self.next_btn.setToolTip("Следующая пара изображений (→)")
        self.next_btn.setStyleSheet("""
            QPushButton {
                background: #9c27b0;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background: #7b1fa2;
            }
        """)
        self.next_btn.clicked.connect(self.navigate_next)
        self.slider_control.addWidget(self.overlay_chk)
        self.slider_control.addWidget(self.fit_to_window_btn)
        self.slider_control.addWidget(self.highlight_diff_btn)
        # self.slider_control.addWidget(self.debug_diff_btn)  # УБРАНО
        self.slider_control.addStretch(1)
        self.slider_control.addWidget(self.prev_btn)
        self.slider_control.addWidget(self.next_btn)
        self.slider_layout.addLayout(self.slider_control)
        self.slider_header = QHBoxLayout()
        self.label_a = QLabel("A: <не выбрано>")
        self.label_a.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #1976d2;
                padding: 4px;
                background: #e3f2fd;
                border-radius: 4px;
            }
        """)
        self.label_b = QLabel("B: <не выбрано>")
        self.label_b.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #388e3c;
                padding: 4px;
                background: #e8f5e8;
                border-radius: 4px;
            }
        """)
        self.slider_header.addWidget(self.label_a)
        self.slider_header.addStretch(1)
        self.slider_header.addWidget(self.label_b)
        self.slider_layout.addLayout(self.slider_header)
        
        # Метка с процентом различия
        self.diff_percentage_label = QLabel("Различие: --%")
        self.diff_percentage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diff_percentage_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #d32f2f;
                padding: 6px;
                background: #ffebee;
                border-radius: 4px;
                margin: 2px;
            }
        """)
        self.slider_layout.addWidget(self.diff_percentage_label)
        self.slider_reveal = SliderReveal(QPixmap(), QPixmap())
        self.slider_layout.addWidget(self.slider_reveal, 1)
        self.slider_reveal.setVisible(True)
        
        # Добавляем панель управления смещением (временно скрыта)
        if self.output_dir:
            self.alignment_manager = ImageAlignmentManager(self.output_dir)
        else:
            self.alignment_manager = ImageAlignmentManager("")
        self.alignment_control_panel = AlignmentControlPanel(self.alignment_manager)
        self.alignment_control_panel.alignment_changed.connect(self.on_alignment_changed)
        # Временно скрываем панель смещения
        self.alignment_control_panel.setVisible(False)
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
        # --- 🎯 Главный QSplitter: три колонки + слайдер ---
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
        # --- 📑 Tabs ---
        print('before tabs')
        self.tabs = QTabWidget()
        print('after tabs')
        self.tabs.addTab(settings_tab, "Настройки сравнения")
        print('after add settings_tab')
        main_tab = QWidget()
        main_layout = QVBoxLayout()
        
        # Добавляем кнопку скрытия/показа панелей в верхнюю часть
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.toggle_folders_btn)
        top_controls.addStretch(1)
        main_layout.addLayout(top_controls)
        
        # Добавляем основной splitter
        main_layout.addWidget(self.main_splitter)
        main_tab.setLayout(main_layout)
        print('after main_tab layout')
        self.tabs.addTab(main_tab, "Сравнение и Слайдер")
        print('after add main_tab')
        self.setCentralWidget(self.tabs)
        print('after setCentralWidget')
        
        # --- 🎨 Устанавливаем иконку программы ---
        try:
            icon_path = "imgdiff_icon.ico"
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                print("✅ Иконка программы установлена")
            else:
                print("⚠️ Файл иконки не найден: imgdiff_icon.ico")
        except Exception as e:
            print(f"⚠️ Не удалось установить иконку: {e}")
        
        # --- 🎨 Современный стиль ---
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
        # --- 📊 Status Bar ---
        self.progress_bar = QProgressBar()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()
        print('after status bar')
        # --- 🔗 Connections ---
        self.grp_a.dir_btn.clicked.connect(lambda: self.load_files(self.grp_a, 'A'))
        self.grp_b.dir_btn.clicked.connect(lambda: self.load_files(self.grp_b, 'B'))
        self.grp_a.table.itemSelectionChanged.connect(self.update_slider)
        self.grp_b.table.itemSelectionChanged.connect(self.update_slider)
        self.grp_a.table.itemDoubleClicked.connect(self.open_table_image)
        self.grp_b.table.itemDoubleClicked.connect(self.open_table_image)
        
        # Включаем контекстное меню для таблиц
        self.grp_a.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grp_b.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grp_a.table.customContextMenuRequested.connect(self.show_context_menu_a)
        self.grp_b.table.customContextMenuRequested.connect(self.show_context_menu_b)
        self.result_table.itemSelectionChanged.connect(self.on_result_selection_changed)
        
        # Подключаем обновление состояния кнопки сохранения
        self.radio_all.toggled.connect(self.update_save_button_state)
        self.radio_sel.toggled.connect(self.update_save_button_state)
        self.grp_a.table.itemSelectionChanged.connect(self.update_save_button_state)
        self.grp_b.table.itemSelectionChanged.connect(self.update_save_button_state)
        
        # Подключаем события выбора в таблицах исключений
        self.exclude_a_table.itemSelectionChanged.connect(self.update_restore_buttons_state)
        self.exclude_b_table.itemSelectionChanged.connect(self.update_restore_buttons_state)
        
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
            
            # Инициализируем менеджер смещения для новой папки (временно скрыт)
            self.alignment_manager = ImageAlignmentManager(dir_path)
            if self.alignment_control_panel:
                self.alignment_control_panel.alignment_manager = self.alignment_manager
                # Временно скрываем панель смещения
                self.alignment_control_panel.setVisible(False)
            
            self.load_results_from_output_dir()
            # Обновляем состояние кнопок сохранения и подсветки
            self.update_save_button_state()

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
            # Инвалидируем кэш overlay при изменении цвета
            if hasattr(self, 'slider_reveal'):
                self.slider_reveal.invalidate_overlay_cache()

    def choose_add_color(self):
        col = QColorDialog.getColor(self.add_color, self, "Выберите цвет для добавленного")
        if col.isValid():
            self.add_color = col
            self.add_color_btn.setText(f"Цвет добавленного: {col.name()}")
            self.add_color_btn.setStyleSheet(f"background:{col.name()}")
            # Инвалидируем кэш overlay при изменении цвета
            if hasattr(self, 'slider_reveal'):
                self.slider_reveal.invalidate_overlay_cache()

    def choose_match_color(self):
        col = QColorDialog.getColor(self.match_color, self, "Выберите цвет для совпадающих линий")
        if col.isValid():
            self.match_color = col
            self.match_color_btn.setText(f"Цвет совпадений: {col.name()}")
            self.match_color_btn.setStyleSheet(f"background:{col.name()}; color:white")
            # Инвалидируем кэш overlay при изменении цвета
            if hasattr(self, 'slider_reveal'):
                self.slider_reveal.invalidate_overlay_cache()

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
        
        # Обновляем состояние кнопок после загрузки файлов
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
                if row is None:
                    row = self.result_table.rowCount()
                    self.result_table.insertRow(row)
                    self.result_table.setItem(row, 0, QTableWidgetItem(out_name))
                    self.result_table.setItem(row, 2, QTableWidgetItem(str(out_path)))
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
            if row is None:
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                self.result_table.setItem(row, 0, QTableWidgetItem(out_name))
                self.result_table.setItem(row, 2, QTableWidgetItem(str(out_path)))
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
        
        # Обновляем состояние кнопок после завершения сравнения
        self.update_save_button_state()

    def add_result(self, name, status, path):
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        self.result_table.setItem(row, 0, QTableWidgetItem(name))
        self.result_table.setItem(row, 1, QTableWidgetItem(status))
        self.result_table.setItem(row, 2, QTableWidgetItem(path))

    def _ensure_result_row(self, name: str, path: str) -> int:
        for r in range(self.result_table.rowCount()):
            item = self.result_table.item(r, 0)
            if item and item.text() == name:
                if not self.result_table.item(r, 2):
                    self.result_table.setItem(r, 2, QTableWidgetItem(path))
                return r
        r = self.result_table.rowCount()
        self.result_table.insertRow(r)
        self.result_table.setItem(r, 0, QTableWidgetItem(name))
        self.result_table.setItem(r, 1, QTableWidgetItem(""))
        self.result_table.setItem(r, 2, QTableWidgetItem(path))
        return r

    def compare_parallel(self):
        files_a = self.grp_a.selected_files() if self.radio_sel.isChecked() else self.grp_a.all_files()
        files_b = self.grp_b.selected_files() if self.radio_sel.isChecked() else self.grp_b.all_files()
        if len(files_a) != len(files_b) or not files_a:
            QMessageBox.warning(self, "Несовпадение выбора", "Выберите одинаковое число файлов в обоих списках.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Нет папки вывода", "Укажите директорию для результатов.")
            return

        import os as _os
        import gc

        self.batch_total = len(files_a)
        self.batch_done = 0
        self.batch_ok = 0
        self.batch_equal = 0
        self.batch_err = 0

        self.progress_bar.setMaximum(self.batch_total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.compare_btn.setEnabled(False)

        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        all_result_files = [
            str(Path(self.output_dir) / f)
            for f in sorted(_os.listdir(self.output_dir))
            if Path(f).suffix.lower() in exts
        ]
        self.result_table.setRowCount(0)
        for f in all_result_files:
            name = _os.path.basename(f)
            self._ensure_result_row(name, f)

        fuzz = self.fuzz_spin.value()
        thick = self.thick_spin.value()
        match_tolerance = self.match_tolerance_spin.value()
        gamma = self.gamma_spin.value()
        morph_open = self.noise_chk.isChecked()
        min_area = self.min_area_spin.value()
        debug = self.debug_chk.isChecked()
        use_ssim = self.ssim_chk.isChecked()
        del_color_bgr = (self.color.blue(), self.color.green(), self.color.red())
        add_color_bgr = (self.add_color.blue(), self.add_color.green(), self.add_color.red())
        match_color_bgr = (self.match_color.blue(), self.match_color.green(), self.match_color.red())

        for a, b in zip(files_a, files_b):
            out_name = f"{Path(a).stem}__vs__{Path(b).stem}_outline.png"
            out_path = Path(self.output_dir) / out_name
            self._ensure_result_row(out_name, str(out_path))
            self.progress_bar.setFormat(f"Обработка: {Path(a).name} vs {Path(b).name}")
            params = {
                'fuzz': fuzz,
                'thick': thick,
                'match_tolerance': match_tolerance,
                'gamma': gamma,
                'morph_open': morph_open,
                'min_area': min_area,
                'debug': debug,
                'use_ssim': use_ssim,
                'del_color_bgr': del_color_bgr,
                'add_color_bgr': add_color_bgr,
                'match_color_bgr': match_color_bgr,
                'output_dir': self.output_dir,
                'out_name': out_name,
            }
            worker = CompareWorker(a, b, out_path, params)
            worker.signals.finished.connect(self._on_worker_finished)
            self.threadpool.start(worker)

    def _on_worker_finished(self, out_name: str, out_path: str, code: int, error_message: str):
        if code == 1:
            status = "OK"
            self.batch_ok += 1
        elif code == 0:
            status = "Equal"
            self.batch_equal += 1
        else:
            status = f"Error{(': ' + error_message) if error_message else ''}"
            self.batch_err += 1

        row = self._ensure_result_row(out_name, out_path)
        self.result_table.setItem(row, 1, QTableWidgetItem(status))

        self.batch_done += 1
        self.progress_bar.setValue(self.batch_done)
        QApplication.processEvents()

        if self.batch_done >= self.batch_total:
            self.progress_bar.hide()
            self.progress_bar.setFormat("")
            gc.collect()
            message = (
                "Сравнение завершено!\n\n"
                f"Обработано пар: {self.batch_total}\n"
                f"Успешно: {self.batch_ok}\n"
                f"Равны: {self.batch_equal}\n"
                f"Ошибки: {self.batch_err}\n\n"
                f"Результаты: {self.output_dir}"
            )
            QMessageBox.information(self, "Сравнение завершено", message)
            self.compare_btn.setEnabled(True)
            self.update_save_button_state()

    def _ensure_result_row(self, name: str, path: str) -> int:
        for r in range(self.result_table.rowCount()):
            item = self.result_table.item(r, 0)
            if item and item.text() == name:
                if not self.result_table.item(r, 2):
                    self.result_table.setItem(r, 2, QTableWidgetItem(path))
                return r
        r = self.result_table.rowCount()
        self.result_table.insertRow(r)
        self.result_table.setItem(r, 0, QTableWidgetItem(name))
        self.result_table.setItem(r, 1, QTableWidgetItem(""))
        self.result_table.setItem(r, 2, QTableWidgetItem(path))
        return r

    def run_outline(self, left, right, out_path, fuzz, thick, color_hex, match_tolerance, match_color):
        old = fast_cv2_imread(str(left))
        new = fast_cv2_imread(str(right))
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
        
        # Быстрый предфильтр: на даунскейле проверяем отличия
        try:
            ratio = quick_diff_ratio(old, new, max_side=256, thr=5)
            if ratio < 0.001:
                # Почти равны — рано выходим без тяжелого diff
                del old, new
                return 0
        except Exception:
            pass

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
        success = cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not success:
            raise Exception(f"Не удалось сохранить результат в {out_path}")
        
        # Освобождаем память
        del old, new, overlay
        
        return 1 if meta['diff_pixels'] > 0 else 0

    def update_slider(self):
        # Используем новую логику с учетом смещения (временно отключено)
        self.update_slider_with_alignment()
        # Обновляем состояние кнопок
        self.update_save_button_state()

    def update_slider_pair(self):
        # этот метод больше не нужен, но обновляем состояние кнопок
        self.update_save_button_state()

    def update_slider_overlay_mode(self):
        self.slider_reveal.setOverlayMode(self.overlay_chk.isChecked())
        # Обновляем состояние кнопок сохранения и подсветки
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
        try:
            if item is not None and hasattr(item, 'row'):
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
        except Exception as e:
            logging.error(f"Ошибка при открытии изображения: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть изображение: {str(e)}")

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
                # Обновляем состояние кнопок
                self.update_save_button_state()
            else:
                # Fallback к cv2 если QPixmap не смог загрузить
                img = fast_cv2_imread(img_path)
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
                    # Обновляем состояние кнопок
                    self.update_save_button_state()
                else:
                    self.result_image_view.setPixmap(QPixmap())
                    self.current_result_index = -1
                    # Обновляем состояние кнопок
                    self.update_save_button_state()
        except Exception as e:
            logging.error(f"Ошибка загрузки превью: {e}")
            self.result_image_view.setPixmap(QPixmap())
            self.current_result_index = -1
            # Обновляем состояние кнопок
            self.update_save_button_state()

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
        # Обновляем состояние кнопок
        self.update_save_button_state()

    def keyPressEvent(self, e):
        """Обработка нажатий клавиш для навигации"""
        if self.tabs.currentWidget() == self.slider_widget:
            if not self.overlay_chk.isChecked():
                # В режиме результата - навигация по результатам
                if e.key() == Qt.Key.Key_Right:
                    self.navigate_result(1)
                    return
                elif e.key() == Qt.Key.Key_Left:
                    self.navigate_result(-1)
                    return
            else:
                # В режиме overlay - навигация по парам изображений
                if e.key() == Qt.Key.Key_Right:
                    self.navigate_tables(1)
                    return
                elif e.key() == Qt.Key.Key_Left:
                    self.navigate_tables(-1)
                    return
        super().keyPressEvent(e)

    def navigate_previous(self):
        """Переход к предыдущей паре изображений"""
        self.navigate_tables(-1)
    
    def navigate_next(self):
        """Переход к следующей паре изображений"""
        self.navigate_tables(1)
    
    def navigate_tables(self, delta):
        """Перемещает выделение между парами изображений в таблицах A и B"""
        # Получаем количество строк в обеих таблицах
        rows_a = self.grp_a.table.rowCount()
        rows_b = self.grp_b.table.rowCount()
        
        if rows_a == 0 or rows_b == 0:
            return  # Нет изображений для навигации
        
        # Получаем текущие выделенные строки
        curr_a = self.grp_a.table.currentRow()
        curr_b = self.grp_b.table.currentRow()
        
        # Определяем, какая таблица активна (имеет выделение)
        if curr_a >= 0 and curr_b >= 0:
            # Обе таблицы имеют выделение - перемещаем обе
            new_a = curr_a + delta
            new_b = curr_b + delta
        elif curr_a >= 0:
            # Только таблица A имеет выделение
            new_a = curr_a + delta
            new_b = new_a  # Синхронизируем с A
        elif curr_b >= 0:
            # Только таблица B имеет выделение
            new_b = curr_b + delta
            new_a = new_b  # Синхронизируем с B
        else:
            # Нет выделения - начинаем с начала или конца
            if delta > 0:
                new_a = 0
                new_b = 0
            else:
                new_a = rows_a - 1
                new_b = rows_b - 1
        
        # Применяем циклическую навигацию
        new_a = new_a % rows_a
        new_b = new_b % rows_b
        
        # Применяем выделение
        self.grp_a.table.selectRow(new_a)
        self.grp_b.table.selectRow(new_b)
        
        # Обновляем слайдер с новым выделением
        self.update_slider()
        
        # Показываем информацию о текущей паре
        self.show_navigation_info(new_a, new_b)

    def show_navigation_info(self, row_a, row_b):
        """Показывает информацию о текущей паре изображений"""
        rows_a = self.grp_a.table.rowCount()
        rows_b = self.grp_b.table.rowCount()
        
        if rows_a > 0 and rows_b > 0:
            # Обновляем заголовки слайдера с информацией о позиции
            file_a_name = self.grp_a.table.item(row_a, 0).text() if row_a < rows_a else "N/A"
            file_b_name = self.grp_b.table.item(row_b, 0).text() if row_b < rows_b else "N/A"
            
            self.label_a.setText(f"A: {file_a_name} ({row_a + 1}/{rows_a})")
            self.label_b.setText(f"B: {file_b_name} ({row_b + 1}/{rows_b})")
            
            # Показываем краткое уведомление в статусной строке
            self.statusBar().showMessage(f"Пара {row_a + 1}/{rows_a}: {file_a_name} ↔ {file_b_name}", 2000)
            
            # Обновляем состояние кнопок
            self.update_save_button_state()

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
        
        # Сохраняем списки исключений
        self.settings.setValue("excluded_files_a", self.get_excluded_files_a())
        self.settings.setValue("excluded_files_b", self.get_excluded_files_b())

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
        
        # Инициализируем менеджер смещения если есть папка вывода (временно скрыт)
        if self.output_dir:
            self.alignment_manager = ImageAlignmentManager(self.output_dir)
            if self.alignment_control_panel:
                self.alignment_control_panel.alignment_manager = self.alignment_manager
                # Временно скрываем панель смещения
                self.alignment_control_panel.setVisible(False)
        
        # Обновляем состояние кнопок после восстановления состояния
        self.update_save_button_state()
        
        # Восстанавливаем списки исключений
        self.restore_excluded_files_lists()

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
                img = fast_cv2_imread(path)
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
                img_a_cv = fast_cv2_imread(a)
                img_b_cv = fast_cv2_imread(b)
                
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
        
        # Обновляем состояние кнопок после завершения сохранения
        self.update_save_button_state()

    def update_save_button_state(self):
        """Обновляет состояние кнопки сохранения на основе текущих настроек"""
        # Включаем кнопку сохранения только если overlay включен И есть файлы для обработки
        files_a = self.grp_a.selected_files() if self.radio_sel.isChecked() else self.grp_a.all_files()
        files_b = self.grp_b.selected_files() if self.radio_sel.isChecked() else self.grp_b.all_files()
        has_files = len(files_a) > 0 and len(files_b) > 0 and len(files_a) == len(files_b)
        has_output_dir = bool(self.output_dir)
        overlay_checked = self.overlay_chk.isChecked()
        
        # Упрощенная логика: включаем кнопки если overlay включен
        # Остальные проверки будут выполнены в save_overlay методе
        should_enable = overlay_checked
        self.save_overlay_btn.setEnabled(should_enable)
        self.highlight_diff_btn.setEnabled(should_enable)
    
    def highlight_differences(self):
        """Подсвечивает места различий мигающим кругом на 3 секунды (асинхронная версия)"""
        try:
            if not self.overlay_chk.isChecked():
                return
                
            # Получаем текущие изображения
            if not hasattr(self.slider_reveal, 'pixmap_a') or not hasattr(self.slider_reveal, 'pixmap_b'):
                return
                
            if self.slider_reveal.pixmap_a.isNull() or self.slider_reveal.pixmap_b.isNull():
                return
            
            # Показываем индикатор загрузки с дополнительной информацией
            img_a = self.slider_reveal.pixmap_a.toImage()
            img_b = self.slider_reveal.pixmap_b.toImage()
            
            self.statusBar().showMessage(f"Анализируем различия... Размеры: {img_a.width()}x{img_a.height()} vs {img_b.width()}x{img_b.height()}", 2000)
            self.highlight_diff_btn.setEnabled(False)  # Блокируем кнопку во время обработки
            
            # Запускаем анализ различий в отдельном потоке через QTimer
            # Это предотвращает зависание UI
            QTimer.singleShot(10, self.create_difference_highlight_animation)
            
        except Exception as e:
            # Обработка ошибок для предотвращения зависания
            logging.error(f"Ошибка при запуске подсветки различий: {e}")
            self.statusBar().showMessage("Ошибка при запуске подсветки", 3000)
            self.highlight_diff_btn.setEnabled(True)  # Разблокируем кнопку
            # self.debug_diff_btn.setEnabled(True)  # Разблокируем кнопку отладки (УБРАНО)
    
    def create_difference_highlight_animation(self):
        """Создает простую подсветку различий прозрачными кругами"""
        try:
            # Получаем изображения
            img_a = self.slider_reveal.pixmap_a.toImage()
            img_b = self.slider_reveal.pixmap_b.toImage()
            
            # Проверяем размеры изображений
            if img_a.width() != img_b.width() or img_a.height() != img_b.height():
                # Если размеры разные, приводим к общему размеру
                max_width = max(img_a.width(), img_b.width())
                max_height = max(img_a.height(), img_b.height())
                img_a = img_a.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_b = img_b.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Простой алгоритм поиска различий без numpy (предотвращает вылеты)
            diff_centers = self.find_differences_simple(img_a, img_b)
            
            # Отладочная информация
            logging.info(f"Найдено центров различий: {len(diff_centers)}")
            if diff_centers:
                logging.info(f"Первые 3 центра: {diff_centers[:3]}")
            
            # Вычисляем процент различия для нового алгоритма
            if diff_centers:
                # Расчет процента на основе количества найденных различий
                total_area = img_a.width() * img_a.height()
                
                # Каждый центр различий представляет область примерно 80x80 пикселей
                # Оптимальный размер для точного расчета
                circle_area = 80 * 80
                estimated_diff_pixels = len(diff_centers) * circle_area
                
                diff_percentage = min((estimated_diff_pixels / total_area) * 100, 100)
                similarity_percentage = 100 - diff_percentage
                
                # Обновляем метку с процентом
                self.diff_percentage_label.setText(f"Различие: {diff_percentage:.1f}% (Сходство: {similarity_percentage:.1f}%)")
                
                # Показываем дополнительную информацию
                self.statusBar().showMessage(f"Найдено {len(diff_centers)} областей различий", 2000)
                
                # Логируем результат
                logging.info(f"Процент различия: {diff_percentage:.1f}%, центров: {len(diff_centers)}")
            else:
                self.diff_percentage_label.setText("Различие: 0.0% (Сходство: 100.0%)")
                self.statusBar().showMessage("Различия не найдены - проверьте цвет в настройках", 2000)
                
                # Логируем отсутствие различий
                logging.warning("Различия не найдены - возможно, цвет в настройках не совпадает с цветом на чертеже")
            
            # Создаем простую подсветку кругами
            self.create_simple_highlight_circles(diff_centers)
            
        except Exception as e:
            # Обработка ошибок для предотвращения зависания
            logging.error(f"Ошибка при создании подсветки различий: {e}")
            self.diff_percentage_label.setText("Ошибка подсветки")
            self.statusBar().showMessage(f"Ошибка подсветки: {str(e)}", 3000)
        finally:
            # Разблокируем кнопку подсветки в любом случае
            self.highlight_diff_btn.setEnabled(True)
            # self.debug_diff_btn.setEnabled(True)  # УБРАНО
    
    def find_differences_simple(self, img_a, img_b):
        """ПРОСТОЙ и стабильный поиск различий"""
        try:
            width = img_a.width()
            height = img_a.height()
            
            logging.info(f"Ищем различия между изображениями A и B")
            logging.info(f"Размер изображения: {width}x{height}")
            
            # УСИЛЕННЫЙ алгоритм поиска различий
            step = 5  # Меньший шаг = больше различий, но медленнее
            color_diff_threshold = 20  # Более низкий порог = более чувствительно
            
            # Список всех найденных различий
            all_differences = []
            
            # Счетчик для логирования
            diff_count = 0
            
            for y in range(0, height, step):
                for x in range(0, width, step):
                    try:
                        # Получаем цвета пикселей из обоих изображений
                        color_a = img_a.pixelColor(x, y)
                        color_b = img_b.pixelColor(x, y)
                        
                        # Вычисляем разницу между цветами
                        color_diff = (
                            abs(color_a.red() - color_b.red()) +
                            abs(color_a.green() - color_b.green()) +
                            abs(color_a.blue() - color_b.blue())
                        ) / 3  # Среднее отклонение по RGB
                        
                        # Более чувствительный порог для поиска различий
                        if color_diff > color_diff_threshold:
                            all_differences.append((x, y, color_diff))
                            diff_count += 1
                            
                            # Логируем только первые несколько различий
                            if diff_count <= 5:
                                logging.info(f"🎯 Найдено различие в ({x}, {y}): разница={color_diff:.1f}")
                            
                            # Увеличенный лимит для большего количества различий
                            if len(all_differences) >= 500:  # Увеличиваем лимит для большего покрытия
                                break
                    except Exception as e:
                        # Пропускаем проблемные пиксели
                        continue
                
                if len(all_differences) >= 500:
                    break
            
            logging.info(f"Найдено различий: {len(all_differences)}")
            
            # УЛУЧШЕННАЯ группировка различий
            centers = []
            
            # Сортируем различия по силе различия (большие различия сначала)
            all_differences.sort(key=lambda x: x[2], reverse=True)
            
            # Берем различия с наибольшей разницей
            for i, (x, y, color_diff) in enumerate(all_differences):
                # Проверяем, не слишком ли близко к уже выбранным центрам
                too_close = False
                for center_x, center_y in centers:
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if distance < 80:  # Минимальное расстояние между центрами
                        too_close = True
                        break
                
                if not too_close:
                    centers.append((x, y))
                    if len(centers) >= 20:  # Увеличиваем количество кружочков
                        break
            
            # ДОПОЛНИТЕЛЬНАЯ ОТЛАДКА: Проверяем, какие координаты в видимой области
            logging.info("🔍 Анализ координат различий:")
            visible_count = 0
            for x, y in centers:
                # Проверяем, находится ли координата в пределах слайдера
                if 0 <= x <= 1779 and 0 <= y <= 1308:  # Примерные размеры слайдера
                    visible_count += 1
                    logging.info(f"  ✅ ({x}, {y}) - в видимой области")
                else:
                    logging.info(f"  ❌ ({x}, {y}) - вне видимой области")
            
            logging.info(f"📊 Из {len(centers)} центров различий в видимой области: {visible_count}")
            
            logging.info(f"Найдено центров различий: {len(centers)}")
            return centers
            
        except Exception as e:
            # Обработка ошибок для предотвращения зависания
            logging.error(f"Ошибка при поиске различий: {e}")
            return []
    
    def group_close_differences(self, differences):
        """Улучшенная группировка различий с правильным алгоритмом"""
        try:
            if not differences:
                return []
            
            # Структура данных: (x, y, color_diff)
            # Сортируем различия по отклонению цвета (большие различия сначала)
            differences.sort(key=lambda x: x[2], reverse=True)
            
            # УЛУЧШЕННАЯ группировка различий
            groups = []
            min_distance = 80  # Уменьшаем расстояние для более плотного покрытия
            
            for x, y, color_diff in differences:
                # Проверяем, можно ли добавить к существующей группе
                added_to_group = False
                
                for group in groups:
                    group_x, group_y, group_count = group
                    # Вычисляем расстояние до центра группы
                    distance = ((x - group_x) ** 2 + (y - group_y) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        # Добавляем к существующей группе
                        # Обновляем центр группы (среднее арифметическое)
                        total_x = group_x * group_count + x
                        total_y = group_y * group_count + y
                        new_count = group_count + 1
                        group[0] = total_x / new_count
                        group[1] = total_y / new_count
                        group[2] = new_count
                        added_to_group = True
                        break
                
                if not added_to_group:
                    # Создаем новую группу
                    groups.append([x, y, 1])
                
                # Увеличиваем лимит групп для лучшего покрытия
                if len(groups) >= 25:  # Больше групп = больше кружочков
                    break
            
            # Преобразуем группы в центры
            centers = []
            for group_x, group_y, count in groups:
                centers.append((int(group_x), int(group_y)))
                logging.info(f"📊 Группа различий: центр=({int(group_x)}, {int(group_y)}), количество пикселей={count}")
            
            return centers
            
        except Exception as e:
            logging.error(f"Ошибка при группировке различий: {e}")
            # Fallback: возвращаем первые различия без группировки
            return [(x, y) for x, y, _ in differences[:10]]
    
    def find_difference_centers(self, diff_mask):
        """Находит центры областей различий для подсветки (устаревший метод)"""
        # Оставляем для совместимости, но не используем
        return []
    
    def start_highlight_animation(self, centers):
        """Запускает анимацию подсветки различий (устаревший метод)"""
        # Оставляем для совместимости, но используем простую версию
        self.create_simple_highlight_circles(centers)
    
    def debug_differences(self):
        """Отладочная функция для анализа различий"""
        try:
            # self.debug_diff_btn.setEnabled(False)  # Блокируем кнопку (УБРАНО)
            
            # Получаем изображения
            img_a = self.slider_reveal.pixmap_a.toImage()
            img_b = self.slider_reveal.pixmap_b.toImage()
            
            if img_a.isNull() or img_b.isNull():
                self.statusBar().showMessage("Изображения не загружены", 3000)
                return
            
            # Показываем информацию о размерах
            size_info = f"Размеры: A={img_a.width()}x{img_a.height()}, B={img_b.width()}x{img_b.height()}"
            
            # Ищем различия
            diff_centers = self.find_differences_simple(img_a, img_b)
            
            # Создаем детальный отчет
            report = f"""
🔍 ОТЧЕТ О РАЗЛИЧИЯХ:

{size_info}

📊 Найдено различий: {len(diff_centers)}
🎯 Координаты центров различий:
"""
            
            for i, (x, y) in enumerate(diff_centers[:10]):  # Показываем первые 10
                report += f"  {i+1}. ({x}, {y})\n"
            
            if len(diff_centers) > 10:
                report += f"  ... и еще {len(diff_centers) - 10} различий\n"
            
            # Показываем отчет в диалоге
            msg = QMessageBox()
            msg.setWindowTitle("🐛 Отладка различий")
            msg.setText(report)
            msg.setDetailedText(f"Полный список координат:\n" + 
                              "\n".join([f"({x}, {y})" for x, y in diff_centers]))
            msg.exec()
            
            self.statusBar().showMessage(f"Отладка завершена. Найдено {len(diff_centers)} различий", 3000)
            
        except Exception as e:
            logging.error(f"Ошибка при отладке различий: {e}")
            self.statusBar().showMessage(f"Ошибка отладки: {str(e)}", 3000)
        finally:
            pass  # self.debug_diff_btn.setEnabled(True)  # УБРАНО
    
    def create_simple_highlight_circles(self, centers):
        """Создает простые круги подсветки без анимации"""
        try:
            if not centers:
                self.statusBar().showMessage("Различия не найдены", 2000)
                return
            
            # Очищаем предыдущие круги подсветки
            if hasattr(self, 'highlight_circles'):
                self.remove_highlight_circles()
            
            # Создаем круги подсветки
            self.highlight_circles = []
            
            # Получаем цвет контура отличий из настроек
            highlight_color = self.color
            color_name = highlight_color.name()
            
            # Логируем используемый цвет для отладки
            logging.info(f"🎨 Цвет подсветки: {color_name} (RGB: {highlight_color.red()}, {highlight_color.green()}, {highlight_color.blue()})")
            
            # ПРОВЕРЯЕМ: Если цвет не зеленый, принудительно используем зеленый для подсветки
            if highlight_color.green() < 200 or highlight_color.red() > 100 or highlight_color.blue() > 100:
                logging.warning(f"⚠️ Цвет подсветки не зеленый: {color_name}, принудительно используем зеленый")
                highlight_color = QColor(0, 255, 0)  # Принудительно зеленый
                color_name = highlight_color.name()
                logging.info(f"🎨 Установлен принудительный зеленый цвет: {color_name}")
            
            # Получаем размеры изображений для проверки координат
            img_a = self.slider_reveal.pixmap_a.toImage()
            img_b = self.slider_reveal.pixmap_b.toImage()
            
            logging.info(f"Размеры изображений: A={img_a.width()}x{img_a.height()}, B={img_b.width()}x{img_b.height()}")
            logging.info(f"Размер слайдера: {self.slider_reveal.width()}x{self.slider_reveal.height()}")
            
            circles_created = 0
            circles_shown = 0
            
            for i, (center_x, center_y) in enumerate(centers):
                logging.info(f"🔍 Обрабатываем центр {i+1}/{len(centers)}: ({center_x}, {center_y})")
                # Создаем круг подсветки как дочерний элемент ГЛАВНОГО ОКНА для лучшего Z-order
                circle = QLabel(self)
                circle.setFixedSize(120, 120)  # УВЕЛИЧИВАЕМ размер для лучшей видимости
                
                circles_created += 1
                
                # УПРОЩАЕМ стили - делаем кружочки ПОЛУПРОЗРАЧНЫМИ (50%)
                circle.setStyleSheet(f"""
                    QLabel {{
                        background: rgba(0, 255, 0, 0.5);
                        border: 10px solid rgba(0, 0, 0, 0.5);
                        border-radius: 60px;
                    }}
                """)
                
                # Принудительно устанавливаем атрибуты для отображения
                circle.setAttribute(Qt.WA_TransparentForMouseEvents, False)
                circle.setAttribute(Qt.WA_NoSystemBackground, False)
                circle.raise_()  # Поднимаем наверх
                
                # ПРОСТОЕ позиционирование круга
                try:
                    # ПРАВИЛЬНОЕ позиционирование круга с учетом масштаба и смещения
                    # Получаем параметры слайдера
                    scale = getattr(self.slider_reveal, 'scale', 1.0)
                    offset_x = getattr(self.slider_reveal, 'offset', QPoint(0, 0)).x()
                    offset_y = getattr(self.slider_reveal, 'offset', QPoint(0, 0)).y()
                    
                    # Преобразуем координаты из оригинального изображения в координаты слайдера
                    # center_x и center_y - это координаты в оригинальном изображении (9933x7017)
                    # Нужно преобразовать их в координаты слайдера (1779x1308)
                    pos_x = int(center_x * scale + offset_x - 60)  # Центрируем круг (120/2)
                    pos_y = int(center_y * scale + offset_y - 60)
                    
                    logging.info(f"📍 Позиция круга {i+1}: оригинал=({center_x}, {center_y}), слайдер=({pos_x}, {pos_y})")
                    logging.info(f"   Масштаб: {scale}, Смещение: ({offset_x}, {offset_y})")
                    
                    # Проверяем, что круг находится в пределах слайдера
                    slider_width = self.slider_reveal.width()
                    slider_height = self.slider_reveal.height()
                    
                    if (pos_x >= -120 and pos_x <= slider_width + 120 and 
                        pos_y >= -120 and pos_y <= slider_height + 120):
                        
                        # ПРЕОБРАЗУЕМ координаты слайдера в координаты главного окна
                        slider_pos = self.slider_reveal.mapToGlobal(QPoint(0, 0))
                        main_pos = self.mapFromGlobal(slider_pos)
                        
                        # Финальные координаты в главном окне
                        final_x = main_pos.x() + pos_x
                        final_y = main_pos.y() + pos_y
                        
                        circle.move(final_x, final_y)
                        circle.show()
                        
                        # ПРИНУДИТЕЛЬНО поднимаем на самый верх несколько раз
                        for _ in range(10):
                            circle.raise_()
                        
                        circle.repaint()
                        circle.update()
                        
                        self.highlight_circles.append(circle)
                        circles_shown += 1
                        
                        # Логируем создание круга
                        logging.info(f"✅ Круг {i+1} создан и показан в слайдере=({pos_x}, {pos_y}), главное окно=({final_x}, {final_y})")
                        
                        # Принудительно обновляем главное окно
                        self.repaint()
                        self.update()
                    else:
                        logging.warning(f"❌ Круг вне области слайдера: ({pos_x}, {pos_y}) vs {slider_width}x{slider_height}")
                        
                        # АЛЬТЕРНАТИВА: Пытаемся найти ближайшую видимую позицию
                        logging.info(f"🔄 Пытаемся найти ближайшую видимую позицию для круга {i+1}")
                        
                        # Ограничиваем координаты пределами слайдера
                        adjusted_x = max(60, min(pos_x, slider_width - 60))
                        adjusted_y = max(60, min(pos_y, slider_height - 60))
                        
                        # Проверяем, что скорректированная позиция отличается от исходной
                        if abs(adjusted_x - pos_x) < slider_width and abs(adjusted_y - pos_y) < slider_height:
                            # ПРЕОБРАЗУЕМ координаты слайдера в координаты главного окна
                            slider_pos = self.slider_reveal.mapToGlobal(QPoint(0, 0))
                            main_pos = self.mapFromGlobal(slider_pos)
                            
                            # Финальные координаты в главном окне
                            final_x = main_pos.x() + adjusted_x
                            final_y = main_pos.y() + adjusted_y
                            
                            circle.move(final_x, final_y)
                            circle.show()
                            
                            # ПРИНУДИТЕЛЬНО поднимаем на самый верх несколько раз
                            for _ in range(10):
                                circle.raise_()
                            
                            circle.repaint()
                            circle.update()
                            
                            self.highlight_circles.append(circle)
                            circles_shown += 1
                            
                            logging.info(f"✅ Круг {i+1} скорректирован и показан в слайдере=({adjusted_x}, {adjusted_y}), главное окно=({final_x}, {final_y})")
                            
                            # Принудительно обновляем главное окно
                            self.repaint()
                            self.update()
                        else:
                            logging.warning(f"❌ Не удалось скорректировать позицию для круга {i+1}")
                            circle.deleteLater()
                        
                except Exception as e:
                    logging.error(f"❌ Ошибка позиционирования круга: {e}")
                    circle.deleteLater()
                    continue
            
            logging.info(f"📊 ИТОГО: создано {circles_created} кружочков, показано {circles_shown}")
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Проверяем, что кружочки действительно в списке
            logging.info(f"🔍 Проверка списка кружочков:")
            logging.info(f"   Длина списка: {len(self.highlight_circles)}")
            logging.info(f"   Тип элементов: {[type(circle) for circle in self.highlight_circles]}")
            logging.info(f"   Видимость: {[circle.isVisible() for circle in self.highlight_circles]}")
            logging.info(f"   Размеры: {[f'{circle.width()}x{circle.height()}' for circle in self.highlight_circles]}")
            
                                        # ПРОСТОЙ ТЕСТ: УБРАН - больше не нужен
            # logging.info("🧪 Создаем ПРОСТОЙ тестовый кружочек прямо в слайдере")
            # simple_circle = QLabel(self.slider_reveal)
            # simple_circle.setFixedSize(100, 100)
            # simple_circle.setStyleSheet("""
            #     QLabel {
            #         background: rgba(255, 0, 0, 0.5);
            #         border: 5px solid rgba(0, 0, 0, 0.5);
            #         border-radius: 50px;
            #     }
            # """)
            
            # # Размещаем в левом верхнем углу слайдера
            # simple_circle.move(50, 50)
            # simple_circle.show()
            # simple_circle.raise_()
            
            # self.highlight_circles.append(simple_circle)
            # logging.info("🧪 ПРОСТОЙ красный кружочек создан в (50, 50) слайдера")
            
            # КОММЕНТАРИЙ: Тестовый кружочек убран, так как функция подсветки работает стабильно
            
            # ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ: Обновляем весь слайдер после создания всех кружочков
            if self.highlight_circles:
                logging.info("🔄 Принудительно обновляем весь слайдер")
                self.slider_reveal.repaint()
                self.slider_reveal.update()
                
                # Также обновляем родительский виджет
                if hasattr(self.slider_reveal, 'parent'):
                    parent = self.slider_reveal.parent()
                    if parent:
                        parent.repaint()
                        parent.update()
                        logging.info("🔄 Обновлен родительский виджет")
                
                # ПРОВЕРКА: Проверяем, что кружочки различий действительно видны
                logging.info("🔍 ПРОВЕРКА видимости кружочков различий:")
                for i, circle in enumerate(self.highlight_circles):
                    if i < 10:  # Только кружочки различий (первые 10)
                        is_visible = circle.isVisible()
                        geometry = circle.geometry()
                        logging.info(f"   Кружочек {i+1}: видимый={is_visible}, геометрия={geometry}")
                        
                        # Принудительно показываем каждый кружочек различий
                        if not is_visible:
                            circle.show()
                            logging.info(f"   Кружочек {i+1} принудительно показан")
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Принудительно поднимаем ВСЕ кружочки различий наверх
            logging.info("🚨 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Поднимаем все кружочки различий наверх")
            for i, circle in enumerate(self.highlight_circles):
                # Поднимаем каждый кружочек наверх 10 раз
                for _ in range(10):
                    circle.raise_()
                circle.repaint()
                circle.update()
                logging.info(f"🚨 Кружочек {i+1} принудительно поднят наверх")
            
            if self.highlight_circles:
                # Показываем уведомление в статусной строке
                self.statusBar().showMessage(f"Подсвечено {len(self.highlight_circles)} областей различий", 2000)
                
                # ТЕСТ: ОГРОМНЫЙ красный кружочек УБРАН - больше не нужен
                # logging.info("🧪 Создаем ОГРОМНЫЙ тестовый кружочек для проверки видимости")
                # huge_circle = QLabel(self.slider_reveal)
                # huge_circle.setFixedSize(200, 200)  # Огромный размер
                # huge_circle.setStyleSheet("""
                #     QLabel {
                #         background: rgba(255, 0, 0, 0.5);
                #         border: 10px solid rgba(0, 0, 0, 0.5);
                #         border-radius: 100px;
                #     }
                # """)
                
                # # Размещаем в правом верхнем углу слайдера
                # huge_pos_x = self.slider_reveal.width() - 220
                # huge_pos_y = 20
                # huge_circle.move(huge_pos_x, huge_pos_y)
                # huge_circle.show()
                
                # # ПРИНУДИТЕЛЬНО поднимаем на самый верх несколько раз
                # for _ in range(5):
                #     huge_circle.raise_()
                
                # huge_circle.repaint()
                # huge_circle.update()
                
                # self.highlight_circles.append(huge_circle)
                # logging.info(f"🧪 ОГРОМНЫЙ красный кружочек создан в ({huge_pos_x}, {huge_pos_y})")
                
                # ДОПОЛНИТЕЛЬНЫЙ ТЕСТ: Зеленый кружочек в центре УБРАН - больше не нужен
                # logging.info("🧪 Создаем ДОПОЛНИТЕЛЬНЫЙ зеленый кружочек в центре слайдера")
                # center_circle = QLabel(self.slider_reveal)
                # center_circle.setFixedSize(150, 150)  # Средний размер
                # center_circle.setStyleSheet("""
                #     QLabel {
                #         background: rgba(0, 255, 0, 0.5);
                #         border: 8px solid rgba(0, 0, 0, 0.5);
                #         border-radius: 75px;
                #     }
                # """)
                
                # # Размещаем в центре слайдера
                # center_pos_x = (self.slider_reveal.width() - 150) // 2
                # center_pos_y = (self.slider_reveal.height() - 150) // 2
                # center_circle.move(center_pos_x, center_pos_y)
                # center_circle.show()
                
                # # ПРИНУДИТЕЛЬНО поднимаем на самый верх несколько раз
                # for _ in range(10):
                #     center_circle.raise_()
                
                # center_circle.repaint()
                # center_circle.update()
                
                # self.highlight_circles.append(center_circle)
                # logging.info(f"🧪 ДОПОЛНИТЕЛЬНЫЙ зеленый кружочек создан в центре ({center_pos_x}, {center_pos_y})")
                
                # ФИНАЛЬНОЕ ОБНОВЛЕНИЕ: Принудительно обновляем весь интерфейс
                logging.info("🚨 ФИНАЛЬНОЕ ОБНОВЛЕНИЕ: Принудительно обновляем весь интерфейс")
                self.slider_reveal.repaint()
                self.slider_reveal.update()
                self.repaint()
                self.update()
                
                # Принудительно обновляем главное окно несколько раз
                for _ in range(5):
                    self.repaint()
                    self.update()
                    QApplication.processEvents()
                
                # Принудительно обновляем все дочерние виджеты
                for child in self.slider_reveal.findChildren(QLabel):
                    if child in self.highlight_circles:
                        child.raise_()
                        child.repaint()
                        child.update()
                        logging.info(f"🚨 Дочерний виджет {child} принудительно обновлен")
                
                # Принудительно обрабатываем события Qt
                QApplication.processEvents()
                
                # Таймер для автоматического удаления кругов через 3 секунды
                QTimer.singleShot(3000, self.remove_highlight_circles)
            else:
                self.statusBar().showMessage("Не удалось создать круги подсветки", 2000)
                
                # ТЕСТ: Создаем один кружочек в центре для проверки
                logging.info("🧪 Создаем тестовый кружочек в центре слайдера")
                test_circle = QLabel(self.slider_reveal)
                test_circle.setFixedSize(80, 80)
                test_circle.setStyleSheet("""
                    QLabel {
                        background: radial-gradient(circle, #ff000080 0%, #ff000060 30%, #ff000040 60%, #ff000020 100%);
                        border: 4px solid #ff0000;
                        border-radius: 40px;
                    }
                """)
                
                # Размещаем в центре слайдера
                center_x = self.slider_reveal.width() // 2 - 40
                center_y = self.slider_reveal.height() // 2 - 40
                test_circle.move(center_x, center_y)
                test_circle.show()
                
                self.highlight_circles = [test_circle]
                logging.info(f"🧪 Тестовый кружочек создан в центре ({center_x}, {center_y})")
                
                # Удаляем через 5 секунд
                QTimer.singleShot(5000, self.remove_highlight_circles)
                
        except Exception as e:
            # Обработка ошибок для предотвращения зависания
            logging.error(f"Ошибка при создании кругов подсветки: {e}")
            self.statusBar().showMessage("Ошибка при создании кругов подсветки", 3000)
    
    def create_flash_animations(self):
        """Создает анимации мигания для кругов подсветки (устаревший метод)"""
        # Оставляем для совместимости, но не используем
        pass
    
    def remove_highlight_circles(self):
        """Удаляет круги подсветки"""
        try:
            # Удаляем круги подсветки
            if hasattr(self, 'highlight_circles'):
                for circle in self.highlight_circles:
                    circle.deleteLater()
                self.highlight_circles = []
                
                # Показываем уведомление об удалении
                self.statusBar().showMessage("Круги подсветки удалены", 1000)
                
        except Exception as e:
            # Обработка ошибок для предотвращения зависания
            logging.error(f"Ошибка при удалении кругов подсветки: {e}")
    

    
    def toggle_folders_panel(self):
        """Скрывает/показывает панели выбора папок для увеличения рабочего пространства"""
        try:
            # Переключаем видимость панелей
            is_visible = self.splitter.isVisible()
            
            if is_visible:
                # Скрываем панели
                self.splitter.setVisible(False)
                self.toggle_folders_btn.setText("👁️ Показать панели")
                self.toggle_folders_btn.setToolTip("Показать панели выбора папок (Ctrl+H)")
                # Увеличиваем размер слайдера
                self.main_splitter.setSizes([0, 1000])
                self.statusBar().showMessage("Панели скрыты - больше места для работы с изображениями", 2000)
            else:
                # Показываем панели
                self.splitter.setVisible(True)
                self.toggle_folders_btn.setText("👁️ Скрыть панели")
                self.toggle_folders_btn.setToolTip("Скрыть панели выбора папок для увеличения рабочего пространства (Ctrl+H)")
                # Восстанавливаем размеры
                self.main_splitter.setSizes([540, 900])
                self.statusBar().showMessage("Панели показаны", 2000)
                
        except Exception as e:
            logging.error(f"Ошибка при переключении панелей: {e}")
            self.statusBar().showMessage(f"Ошибка: {str(e)}", 3000)
    
    def on_alignment_changed(self, offset_x: int, offset_y: int):
        """Обработчик изменения смещения изображений (временно отключен)"""
        # Обновляем слайдер с новым смещением
        self.update_slider_with_alignment()
    
    def update_slider_with_alignment(self):
        """Обновляет слайдер с учетом смещения изображений (временно отключено)"""
        # ВРЕМЕННО ОТКЛЮЧЕНО: Функциональность смещения изображений будет реализована позже
        # Пока используем простую логику без смещения
        
        files_a = self.grp_a.selected_files()
        files_b = self.grp_b.selected_files()
        file_a = files_a[0] if files_a else None
        file_b = files_b[0] if files_b else None

        self.label_a.setText(f"A: {Path(file_a).name if file_a else '<не выбрано>'}")
        self.label_b.setText(f"B: {Path(file_b).name if file_b else '<не выбрано>'}")

        if file_a and file_b:
            # Загружаем изображения в оригинальном разрешении (без смещения)
            img_a_cv = fast_cv2_imread(file_a)
            img_b_cv = fast_cv2_imread(file_b)
            if img_a_cv is not None and img_b_cv is not None:
                img_a = QPixmap.fromImage(cv2_to_qimage(img_a_cv))
                img_b = QPixmap.fromImage(cv2_to_qimage(img_b_cv))
                img_a.setDevicePixelRatio(1.0)
                img_b.setDevicePixelRatio(1.0)
                
                # ВРЕМЕННО: Не применяем смещение
                # aligned_a, aligned_b = self.alignment_manager.apply_alignment_to_pixmaps(
                #     img_a, img_b, file_a, file_b
                # )
                
                self.slider_reveal.setPixmaps(img_a, img_b)
                self.slider_reveal.setVisible(True)
                
                # ВРЕМЕННО: Не обновляем панель управления смещением
                # if self.alignment_control_panel:
                #     self.alignment_control_panel.set_current_images(file_a, file_b, img_a, img_b)
                
                # Обновляем состояние кнопок сохранения и подсветки
                self.update_save_button_state()
            else:
                self.slider_reveal.setVisible(False)
                self.save_overlay_btn.setEnabled(False)
                self.highlight_diff_btn.setEnabled(False)
        else:
            self.slider_reveal.setVisible(False)
            self.save_overlay_btn.setEnabled(False)
            self.highlight_diff_btn.setEnabled(False)


    # --- 🚫 Методы для работы с исключениями ---
    
    def exclude_a_drag_enter_event(self, event):
        """Обработчик входа в зону перетаскивания для исключений A"""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
    
    def exclude_a_drag_move_event(self, event):
        """Обработчик движения перетаскивания для исключений A"""
        event.acceptProposedAction()
    
    def exclude_a_drop_event(self, event):
        """Обработчик сброса файлов в исключения A"""
        try:
            mime_data = event.mimeData()
            if mime_data.hasUrls():
                urls = mime_data.urls()
                for url in urls:
                    file_path = url.toLocalFile()
                    if os.path.isfile(file_path):
                        self.exclude_file_a(file_path)
            elif mime_data.hasText():
                text = mime_data.text()
                if os.path.isfile(text):
                    self.exclude_file_a(text)
            event.acceptProposedAction()
        except Exception as e:
            logging.error(f"Ошибка при добавлении файла в исключения A: {e}")
    
    def exclude_b_drag_enter_event(self, event):
        """Обработчик входа в зону перетаскивания для исключений B"""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
    
    def exclude_b_drag_move_event(self, event):
        """Обработчик движения перетаскивания для исключений B"""
        event.acceptProposedAction()
    
    def exclude_b_drop_event(self, event):
        """Обработчик сброса файлов в исключения B"""
        try:
            mime_data = event.mimeData()
            if mime_data.hasUrls():
                urls = mime_data.urls()
                for url in urls:
                    file_path = url.toLocalFile()
                    if os.path.isfile(file_path):
                        self.exclude_file_b(file_path)
            elif mime_data.hasText():
                text = mime_data.text()
                if os.path.isfile(text):
                    self.exclude_file_b(text)
            event.acceptProposedAction()
        except Exception as e:
            logging.error(f"Ошибка при добавлении файла в исключения B: {e}")
    
    def exclude_file_a(self, file_path):
        """Исключить файл из папки A"""
        try:
            logging.info(f"Пытаемся исключить файл A: {file_path}")
            
            # Проверяем, что файл есть в основной таблице A
            if not self.is_file_in_table_a(file_path):
                logging.warning(f"Файл не найден в таблице A: {file_path}")
                return
            
            logging.info(f"Файл найден в таблице A, добавляем в исключения")
            
            # Добавляем в таблицу исключений
            self.add_to_exclude_table(self.exclude_a_table, file_path)
            
            # Убираем из основной таблицы
            self.remove_from_table_a(file_path)
            
            # Обновляем состояние кнопки возврата
            self.update_restore_buttons_state()
            
            logging.info(f"Файл исключен из A: {os.path.basename(file_path)}")
            
        except Exception as e:
            logging.error(f"Ошибка при исключении файла A: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось исключить файл: {str(e)}")
    
    def exclude_file_b(self, file_path):
        """Исключить файл из папки B"""
        try:
            # Проверяем, что файл есть в основной таблице B
            if not self.is_file_in_table_b(file_path):
                return
            
            # Добавляем в таблицу исключений
            self.add_to_exclude_table(self.exclude_b_table, file_path)
            
            # Убираем из основной таблицы
            self.remove_from_table_b(file_path)
            
            # Обновляем состояние кнопки возврата
            self.update_restore_buttons_state()
            
            logging.info(f"Файл исключен из B: {os.path.basename(file_path)}")
            
        except Exception as e:
            logging.error(f"Ошибка при исключении файла B: {e}")
    
    def add_to_exclude_table(self, table, file_path):
        """Добавить файл в таблицу исключений"""
        try:
            # Проверяем, что файл еще не в таблице
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item and item.data(Qt.UserRole) == file_path:
                    return  # Файл уже есть
            
            # Добавляем новый файл
            row = table.rowCount()
            table.insertRow(row)
            
            # Создаем элемент с именем файла
            name_item = QTableWidgetItem(os.path.basename(file_path))
            name_item.setData(Qt.UserRole, file_path)  # Сохраняем полный путь
            name_item.setToolTip(file_path)  # Показываем полный путь при наведении
            
            table.setItem(row, 0, name_item)
            
        except Exception as e:
            logging.error(f"Ошибка при добавлении в таблицу исключений: {e}")
    
    def remove_from_table_a(self, file_path):
        """Убрать файл из таблицы A"""
        try:
            table = self.grp_a.table
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item and item.data(Qt.UserRole) == file_path:
                    table.removeRow(row)
                    break
        except Exception as e:
            logging.error(f"Ошибка при удалении из таблицы A: {e}")
    
    def remove_from_table_b(self, file_path):
        """Убрать файл из таблицы B"""
        try:
            table = self.grp_b.table
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item and item.data(Qt.UserRole) == file_path:
                    table.removeRow(row)
                    break
        except Exception as e:
            logging.error(f"Ошибка при удалении из таблицы B: {e}")
    
    def is_file_in_table_a(self, file_path):
        """Проверить, есть ли файл в таблице A"""
        try:
            table = self.grp_a.table
            logging.info(f"Проверяем файл в таблице A: {file_path}")
            logging.info(f"Количество строк в таблице A: {table.rowCount()}")
            
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item:
                    stored_path = item.data(Qt.UserRole)
                    logging.info(f"Строка {row}: {stored_path}")
                    if stored_path == file_path:
                        logging.info(f"Файл найден в строке {row}")
                        return True
                else:
                    logging.warning(f"Строка {row}: элемент не найден")
            
            logging.warning(f"Файл не найден в таблице A: {file_path}")
            return False
        except Exception as e:
            logging.error(f"Ошибка при проверке файла в таблице A: {e}")
            return False
    
    def is_file_in_table_b(self, file_path):
        """Проверить, есть ли файл в таблице B"""
        try:
            table = self.grp_b.table
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item and item.data(Qt.UserRole) == file_path:
                    return True
            return False
        except Exception as e:
            logging.error(f"Ошибка при проверке файла в таблице B: {e}")
            return False
    
    def restore_excluded_files_a(self):
        """Вернуть выбранные файлы из исключений A"""
        try:
            selected_rows = set()
            for item in self.exclude_a_table.selectedItems():
                row = item.row()
                if row not in selected_rows:
                    selected_rows.add(row)
            
            # Обрабатываем строки в обратном порядке, чтобы не сбить индексы
            for row in sorted(selected_rows, reverse=True):
                item = self.exclude_a_table.item(row, 0)
                if item:
                    file_path = item.data(Qt.UserRole)
                    if file_path and os.path.isfile(file_path):
                        # Возвращаем в основную таблицу A
                        self.restore_file_to_table_a(file_path)
                        # Убираем из таблицы исключений
                        self.exclude_a_table.removeRow(row)
            
            # Обновляем состояние кнопки возврата
            self.update_restore_buttons_state()
            
            logging.info(f"Возвращено {len(selected_rows)} файлов из исключений A")
            
        except Exception as e:
            logging.error(f"Ошибка при возврате файлов из исключений A: {e}")
    
    def restore_excluded_files_b(self):
        """Вернуть выбранные файлы из исключений B"""
        try:
            selected_rows = set()
            for item in self.exclude_b_table.selectedItems():
                row = item.row()
                if row not in selected_rows:
                    selected_rows.add(row)
            
            # Обрабатываем строки в обратном порядке, чтобы не сбить индексы
            for row in sorted(selected_rows, reverse=True):
                item = self.exclude_b_table.item(row, 0)
                if item:
                    file_path = item.data(Qt.UserRole)
                    if file_path and os.path.isfile(file_path):
                        # Возвращаем в основную таблицу B
                        self.restore_file_to_table_b(file_path)
                        # Убираем из таблицы исключений
                        self.exclude_b_table.removeRow(row)
            
            # Обновляем состояние кнопки возврата
            self.update_restore_buttons_state()
            
            logging.info(f"Возвращено {len(selected_rows)} файлов из исключений B")
            
        except Exception as e:
            logging.error(f"Ошибка при возврате файлов из исключений B: {e}")
    
    def restore_file_to_table_a(self, file_path):
        """Вернуть файл в таблицу A"""
        try:
            table = self.grp_a.table
            row = table.rowCount()
            table.insertRow(row)
            
            # Создаем элемент с именем файла
            name_item = QTableWidgetItem(os.path.basename(file_path))
            name_item.setData(Qt.UserRole, file_path)
            name_item.setToolTip(file_path)
            
            table.setItem(row, 0, name_item)
            
            # Сортируем таблицу
            self.grp_a.sort_table()
            
        except Exception as e:
            logging.error(f"Ошибка при возврате файла в таблицу A: {e}")
    
    def restore_file_to_table_b(self, file_path):
        """Вернуть файл в таблицу B"""
        try:
            table = self.grp_b.table
            row = table.rowCount()
            table.insertRow(row)
            
            # Создаем элемент с именем файла
            name_item = QTableWidgetItem(os.path.basename(file_path))
            name_item.setData(Qt.UserRole, file_path)
            name_item.setToolTip(file_path)
            
            # Создаем элемент с размером (если есть)
            size_item = QTableWidgetItem()
            if os.path.exists(file_path):
                try:
                    size = os.path.getsize(file_path)
                    size_item.setText(self.format_file_size(size))
                except:
                    size_item.setText("")
            
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, size_item)
            
            # Сортируем таблицу
            self.grp_b.sort_table()
            
        except Exception as e:
            logging.error(f"Ошибка при возврате файла в таблицу B: {e}")
    
    def update_restore_buttons_state(self):
        """Обновить состояние кнопок возврата файлов"""
        try:
            # Кнопка возврата A
            has_selection_a = len(self.exclude_a_table.selectedItems()) > 0
            self.restore_a_btn.setEnabled(has_selection_a)
            
            # Кнопка возврата B
            has_selection_b = len(self.exclude_b_table.selectedItems()) > 0
            self.restore_b_btn.setEnabled(has_selection_b)
            
        except Exception as e:
            logging.error(f"Ошибка при обновлении состояния кнопок возврата: {e}")
    
    def get_excluded_files_a(self):
        """Получить список исключенных файлов A"""
        try:
            files = []
            for row in range(self.exclude_a_table.rowCount()):
                item = self.exclude_a_table.item(row, 0)
                if item:
                    file_path = item.data(Qt.UserRole)
                    if file_path:
                        files.append(file_path)
            return files
        except Exception as e:
            logging.error(f"Ошибка при получении списка исключений A: {e}")
            return []
    
    def get_excluded_files_b(self):
        """Получить список исключенных файлов B"""
        try:
            files = []
            for row in range(self.exclude_b_table.rowCount()):
                item = self.exclude_b_table.item(row, 0)
                if item:
                    file_path = item.data(Qt.UserRole)
                    if file_path:
                        files.append(file_path)
            return files
        except Exception as e:
            logging.error(f"Ошибка при получении списка исключений B: {e}")
            return []
    
    def restore_excluded_files_lists(self):
        """Восстановить списки исключений из настроек"""
        try:
            # Восстанавливаем исключения A
            excluded_a = self.settings.value("excluded_files_a", [])
            if excluded_a:
                for file_path in excluded_a:
                    if os.path.isfile(file_path):
                        self.add_to_exclude_table(self.exclude_a_table, file_path)
                        # Убираем из основной таблицы, если там есть
                        if self.is_file_in_table_a(file_path):
                            self.remove_from_table_a(file_path)
            
            # Восстанавливаем исключения B
            excluded_b = self.settings.value("excluded_files_b", [])
            if excluded_b:
                for file_path in excluded_b:
                    if os.path.isfile(file_path):
                        self.add_to_exclude_table(self.exclude_b_table, file_path)
                        # Убираем из основной таблицы, если там есть
                        if self.is_file_in_table_b(file_path):
                            self.remove_from_table_b(file_path)
            
            # Обновляем состояние кнопок
            self.update_restore_buttons_state()
            
            logging.info(f"Восстановлено {len(excluded_a or [])} исключений A и {len(excluded_b or [])} исключений B")
            
        except Exception as e:
            logging.error(f"Ошибка при восстановлении списков исключений: {e}")
    
    def show_context_menu_a(self, position):
        """Показать контекстное меню для таблицы A"""
        try:
            table = self.grp_a.table
            context_menu = QMenu(self)
            
            # Получаем выбранные строки
            selected_rows = set()
            for item in table.selectedItems():
                if item.column() == 0:  # Только элементы первой колонки
                    selected_rows.add(item.row())
            
            if selected_rows:
                # Действие "Исключить"
                exclude_action = context_menu.addAction("🚫 Исключить из сравнения")
                exclude_action.triggered.connect(lambda: self.exclude_selected_files_a())
                
                # Разделитель
                context_menu.addSeparator()
                
                # Действие "Открыть изображение"
                open_action = context_menu.addAction("👁️ Открыть изображение")
                open_action.triggered.connect(lambda: self.open_table_image_from_context(table, selected_rows))
                
                # Показываем меню
                context_menu.exec_(table.mapToGlobal(position))
                
        except Exception as e:
            logging.error(f"Ошибка при показе контекстного меню A: {e}")
    
    def show_context_menu_b(self, position):
        """Показать контекстное меню для таблицы B"""
        try:
            table = self.grp_b.table
            context_menu = QMenu(self)
            
            # Получаем выбранные строки
            selected_rows = set()
            for item in table.selectedItems():
                if item.column() == 0:  # Только элементы первой колонки
                    selected_rows.add(item.row())
            
            if selected_rows:
                # Действие "Исключить"
                exclude_action = context_menu.addAction("🚫 Исключить из сравнения")
                exclude_action.triggered.connect(lambda: self.exclude_selected_files_b())
                
                # Разделитель
                context_menu.addSeparator()
                
                # Действие "Открыть изображение"
                open_action = context_menu.addAction("👁️ Открыть изображение")
                open_action.triggered.connect(lambda: self.open_table_image_from_context(table, selected_rows))
                
                # Показываем меню
                context_menu.exec_(table.mapToGlobal(position))
                
        except Exception as e:
            logging.error(f"Ошибка при показе контекстного меню B: {e}")
    
    def exclude_selected_files_a(self):
        """Исключить выбранные файлы из таблицы A"""
        try:
            table = self.grp_a.table
            selected_rows = set()
            
            # Собираем выбранные строки
            for item in table.selectedItems():
                if item.column() == 0:  # Только элементы первой колонки
                    selected_rows.add(item.row())
            
            if not selected_rows:
                return
            
            # Обрабатываем строки в обратном порядке, чтобы не сбить индексы
            for row in sorted(selected_rows, reverse=True):
                item = table.item(row, 0)
                if item:
                    file_path = item.data(Qt.UserRole)
                    if file_path and os.path.isfile(file_path):
                        # Исключаем файл
                        self.exclude_file_a(file_path)
                    else:
                        logging.warning(f"Файл не найден или путь некорректен: {file_path}")
                else:
                    logging.warning(f"Не удалось получить элемент для строки {row}")
            
            # Обновляем состояние кнопки возврата
            self.update_restore_buttons_state()
            
            logging.info(f"Исключено {len(selected_rows)} файлов из A")
            
        except Exception as e:
            logging.error(f"Ошибка при исключении файлов A: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось исключить файлы: {str(e)}")
    
    def exclude_selected_files_b(self):
        """Исключить выбранные файлы из таблицы B"""
        try:
            table = self.grp_b.table
            selected_rows = set()
            
            # Собираем выбранные строки
            for item in table.selectedItems():
                if item.column() == 0:  # Только элементы первой колонки
                    selected_rows.add(item.row())
            
            if not selected_rows:
                return
            
            # Обрабатываем строки в обратном порядке, чтобы не сбить индексы
            for row in sorted(selected_rows, reverse=True):
                item = table.item(row, 0)
                if item:
                    file_path = item.data(Qt.UserRole)
                    if file_path and os.path.isfile(file_path):
                        # Исключаем файл
                        self.exclude_file_b(file_path)
                    else:
                        logging.warning(f"Файл не найден или путь некорректен: {file_path}")
                else:
                    logging.warning(f"Не удалось получить элемент для строки {row}")
            
            # Обновляем состояние кнопки возврата
            self.update_restore_buttons_state()
            
            logging.info(f"Исключено {len(selected_rows)} файлов из B")
            
        except Exception as e:
            logging.error(f"Ошибка при исключении файлов B: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось исключить файлы: {str(e)}")
    
    def open_table_image_from_context(self, table, selected_rows):
        """Открыть изображение из контекстного меню"""
        try:
            if not selected_rows:
                return
            
            # Берем первый выбранный файл
            row = min(selected_rows)
            item = table.item(row, 0)
            if item:
                file_path = item.data(Qt.UserRole)
                logging.info(f"Пытаемся открыть файл: {file_path}")
                if file_path and os.path.isfile(file_path):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(file_path)))
                    logging.info(f"Файл успешно открыт: {file_path}")
                else:
                    logging.warning(f"Файл не найден: {file_path}")
                    QMessageBox.warning(self, "Ошибка", f"Файл не найден: {file_path}")
            else:
                logging.warning(f"Не удалось получить элемент для строки {row}")
                QMessageBox.warning(self, "Ошибка", f"Не удалось получить элемент для строки {row}")
        except Exception as e:
            logging.error(f"Ошибка при открытии изображения из контекстного меню: {e}")
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть изображение: {str(e)}")
    
    def format_file_size(self, size_bytes):
        """Форматировать размер файла в читаемый вид"""
        try:
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes // 1024} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes // (1024 * 1024)} MB"
            else:
                return f"{size_bytes // (1024 * 1024 * 1024)} GB"
        except:
            return ""


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

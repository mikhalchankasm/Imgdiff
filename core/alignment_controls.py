from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSpinBox, QComboBox, QGroupBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from typing import List, Tuple, Optional
from core.image_alignment import ImageAlignmentManager


class PointSelectionWidget(QWidget):
    """Виджет для выбора точек на изображении"""
    
    point_selected = pyqtSignal(int, int)  # x, y координаты
    
    def __init__(self, pixmap: QPixmap, title: str, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.title = title
        self.points: List[QPoint] = []
        self.selected_point_index = -1
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)
        
        # Создаем layout
        layout = QVBoxLayout(self)
        
        # Заголовок
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title_label)
        
        # Инструкция
        instruction = QLabel("Кликните для добавления точки\nCtrl+клик для удаления")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet("color: gray; font-size: 9px;")
        layout.addWidget(instruction)
        
        # Список точек
        self.points_label = QLabel("Точки: нет")
        self.points_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.points_label)
    
    def setPixmap(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.update()
    
    def clearPoints(self):
        self.points.clear()
        self.selected_point_index = -1
        self.update_points_label()
        self.update()
    
    def getPoints(self) -> List[Tuple[int, int]]:
        """Возвращает список точек в координатах изображения"""
        if self.pixmap.isNull():
            return []
        
        # Возвращаем точки как есть, так как они уже в координатах изображения
        return [(p.x(), p.y()) for p in self.points]
    
    def update_points_label(self):
        if not self.points:
            self.points_label.setText("Точки: нет")
        else:
            self.points_label.setText(f"Точки: {len(self.points)}")
    
    def paintEvent(self, event):
        if self.pixmap.isNull():
            return
        
        painter = QPainter(self)
        
        # Рисуем изображение с масштабированием
        scaled_pixmap = self.pixmap.scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Центрируем изображение
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x, y, scaled_pixmap)
        
        # Рисуем точки
        if self.points:
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            
            for i, point in enumerate(self.points):
                # Масштабируем координаты точки
                scaled_x = int(point.x() * scaled_pixmap.width() / self.pixmap.width()) + x
                scaled_y = int(point.y() * scaled_pixmap.height() / self.pixmap.height()) + y
                
                # Рисуем круг
                painter.drawEllipse(scaled_x - 5, scaled_y - 5, 10, 10)
                
                # Номер точки
                painter.drawText(scaled_x + 8, scaled_y + 4, str(i + 1))
        
        painter.end()
    
    def mousePressEvent(self, event):
        if self.pixmap.isNull():
            return
        
        if event.button() == Qt.LeftButton:
            # Конвертируем координаты мыши в координаты изображения
            scaled_pixmap = self.pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # Центрируем изображение
            x_offset = (self.width() - scaled_pixmap.width()) // 2
            y_offset = (self.height() - scaled_pixmap.height()) // 2
            
            # Конвертируем координаты мыши в координаты изображения
            mouse_x = event.x() - x_offset
            mouse_y = event.y() - y_offset
            
            if mouse_x < 0 or mouse_y < 0 or mouse_x >= scaled_pixmap.width() or mouse_y >= scaled_pixmap.height():
                return  # Клик вне изображения
            
            # Масштабируем обратно к оригинальному размеру
            x = int(mouse_x * self.pixmap.width() / scaled_pixmap.width())
            y = int(mouse_y * self.pixmap.height() / scaled_pixmap.height())
            
            if event.modifiers() & Qt.ControlModifier:
                # Удаляем ближайшую точку
                if self.points:
                    min_dist = float('inf')
                    min_index = -1
                    
                    for i, point in enumerate(self.points):
                        # Конвертируем координаты точки для сравнения с координатами мыши
                        scaled_x = int(point.x() * scaled_pixmap.width() / self.pixmap.width()) + x_offset
                        scaled_y = int(point.y() * scaled_pixmap.height() / self.pixmap.height()) + y_offset
                        
                        dist = ((scaled_x - event.x()) ** 2 + (scaled_y - event.y()) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            min_index = i
                    
                    if min_dist < 20:  # Порог для удаления
                        del self.points[min_index]
                        self.update_points_label()
                        self.update()
            else:
                # Добавляем новую точку
                self.points.append(QPoint(x, y))
                self.update_points_label()
                self.update()
                self.point_selected.emit(x, y)


class AlignmentControlPanel(QWidget):
    """Панель управления смещением изображений"""
    
    alignment_changed = pyqtSignal(int, int)  # offset_x, offset_y
    
    def __init__(self, alignment_manager: ImageAlignmentManager, parent=None):
        super().__init__(parent)
        self.alignment_manager = alignment_manager
        self.current_file_a = ""
        self.current_file_b = ""
        self.current_pixmap_a = QPixmap()
        self.current_pixmap_b = QPixmap()
        
        self.setup_ui()
        self.update_controls()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Группа управления смещением
        alignment_group = QGroupBox("Смещение изображений")
        alignment_layout = QVBoxLayout(alignment_group)
        
        # Выбор смещаемого изображения
        moving_layout = QHBoxLayout()
        moving_layout.addWidget(QLabel("Смещать:"))
        self.moving_combo = QComboBox()
        self.moving_combo.addItems(["Изображение B", "Изображение A"])
        self.moving_combo.currentIndexChanged.connect(self.on_moving_image_changed)
        moving_layout.addWidget(self.moving_combo)
        alignment_layout.addLayout(moving_layout)
        
        # Кнопки смещения по X
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_left_btn = QPushButton("←")
        self.x_left_btn.setFixedWidth(30)
        self.x_left_btn.clicked.connect(lambda: self.adjust_offset(-1, 0))
        x_layout.addWidget(self.x_left_btn)
        
        self.x_spin = QSpinBox()
        self.x_spin.setRange(-1000, 1000)
        self.x_spin.setSuffix(" px")
        self.x_spin.valueChanged.connect(self.on_offset_changed)
        x_layout.addWidget(self.x_spin)
        
        self.x_right_btn = QPushButton("→")
        self.x_right_btn.setFixedWidth(30)
        self.x_right_btn.clicked.connect(lambda: self.adjust_offset(1, 0))
        x_layout.addWidget(self.x_right_btn)
        alignment_layout.addLayout(x_layout)
        
        # Кнопки смещения по Y
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_up_btn = QPushButton("↑")
        self.y_up_btn.setFixedWidth(30)
        self.y_up_btn.clicked.connect(lambda: self.adjust_offset(0, -1))
        y_layout.addWidget(self.y_up_btn)
        
        self.y_spin = QSpinBox()
        self.y_spin.setRange(-1000, 1000)
        self.y_spin.setSuffix(" px")
        self.y_spin.valueChanged.connect(self.on_offset_changed)
        y_layout.addWidget(self.y_spin)
        
        self.y_down_btn = QPushButton("↓")
        self.y_down_btn.setFixedWidth(30)
        self.y_down_btn.clicked.connect(lambda: self.adjust_offset(0, 1))
        y_layout.addWidget(self.y_down_btn)
        alignment_layout.addLayout(y_layout)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Сброс")
        self.reset_btn.clicked.connect(self.reset_alignment)
        buttons_layout.addWidget(self.reset_btn)
        
        self.save_btn = QPushButton("Сохранить")
        self.save_btn.clicked.connect(self.save_alignment)
        buttons_layout.addWidget(self.save_btn)
        
        alignment_layout.addLayout(buttons_layout)
        layout.addWidget(alignment_group)
        
        # Группа автоматического выравнивания
        auto_group = QGroupBox("Автоматическое выравнивание")
        auto_layout = QVBoxLayout(auto_group)
        
        # Кнопка автоматического выравнивания
        self.auto_align_btn = QPushButton("Авто-выравнивание")
        self.auto_align_btn.clicked.connect(self.auto_align)
        auto_layout.addWidget(self.auto_align_btn)
        
        # Чекбокс для режима выбора точек
        self.point_selection_mode = QCheckBox("Режим выбора точек")
        self.point_selection_mode.toggled.connect(self.on_point_selection_mode_toggled)
        auto_layout.addWidget(self.point_selection_mode)
        
        # Виджеты выбора точек
        points_layout = QHBoxLayout()
        
        self.points_widget_a = PointSelectionWidget(QPixmap(), "Точки A")
        self.points_widget_a.point_selected.connect(self.on_point_selected_a)
        points_layout.addWidget(self.points_widget_a)
        
        self.points_widget_b = PointSelectionWidget(QPixmap(), "Точки B")
        self.points_widget_b.point_selected.connect(self.on_point_selected_b)
        points_layout.addWidget(self.points_widget_b)
        
        auto_layout.addLayout(points_layout)
        
        # Кнопка расчета по точкам
        self.calculate_from_points_btn = QPushButton("Рассчитать по точкам")
        self.calculate_from_points_btn.clicked.connect(self.calculate_from_points)
        self.calculate_from_points_btn.setEnabled(False)
        auto_layout.addWidget(self.calculate_from_points_btn)
        
        layout.addWidget(auto_group)
        
        # Изначально скрываем виджеты выбора точек
        self.points_widget_a.setVisible(False)
        self.points_widget_b.setVisible(False)
    
    def set_current_images(self, file_a: str, file_b: str, pixmap_a: QPixmap, pixmap_b: QPixmap):
        """Устанавливает текущие изображения для работы"""
        self.current_file_a = file_a
        self.current_file_b = file_b
        self.current_pixmap_a = pixmap_a
        self.current_pixmap_b = pixmap_b
        
        # Обновляем виджеты выбора точек
        self.points_widget_a.setPixmap(pixmap_a)
        self.points_widget_b.setPixmap(pixmap_b)
        
        # Загружаем сохраненные настройки
        self.load_current_alignment()
    
    def load_current_alignment(self):
        """Загружает настройки смещения для текущих изображений"""
        if not self.current_file_a or not self.current_file_b:
            return
        
        settings = self.alignment_manager.get_alignment(self.current_file_a, self.current_file_b)
        
        # Блокируем сигналы для предотвращения рекурсии
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.moving_combo.blockSignals(True)
        
        self.x_spin.setValue(settings.offset_x)
        self.y_spin.setValue(settings.offset_y)
        self.moving_combo.setCurrentIndex(0 if settings.moving_image == "B" else 1)
        
        # Разблокируем сигналы
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.moving_combo.blockSignals(False)
        
        self.update_controls()
    
    def update_controls(self):
        """Обновляет состояние элементов управления"""
        has_images = bool(self.current_file_a and self.current_file_b)
        
        self.x_spin.setEnabled(has_images)
        self.y_spin.setEnabled(has_images)
        self.x_left_btn.setEnabled(has_images)
        self.x_right_btn.setEnabled(has_images)
        self.y_up_btn.setEnabled(has_images)
        self.y_down_btn.setEnabled(has_images)
        self.reset_btn.setEnabled(has_images)
        self.save_btn.setEnabled(has_images)
        self.auto_align_btn.setEnabled(has_images)
        self.moving_combo.setEnabled(has_images)
    
    def adjust_offset(self, delta_x: int, delta_y: int):
        """Корректирует смещение на заданную величину"""
        new_x = self.x_spin.value() + delta_x
        new_y = self.y_spin.value() + delta_y
        
        self.x_spin.setValue(new_x)
        self.y_spin.setValue(new_y)
    
    def on_offset_changed(self):
        """Обработчик изменения смещения"""
        offset_x = self.x_spin.value()
        offset_y = self.y_spin.value()
        self.alignment_changed.emit(offset_x, offset_y)
    
    def on_moving_image_changed(self, index: int):
        """Обработчик изменения смещаемого изображения"""
        # Пересчитываем смещение при смене смещаемого изображения
        self.on_offset_changed()
    
    def reset_alignment(self):
        """Сбрасывает смещение"""
        self.x_spin.setValue(0)
        self.y_spin.setValue(0)
    
    def save_alignment(self):
        """Сохраняет текущие настройки смещения"""
        if not self.current_file_a or not self.current_file_b:
            return
        
        offset_x = self.x_spin.value()
        offset_y = self.y_spin.value()
        moving_image = "A" if self.moving_combo.currentIndex() == 1 else "B"
        
        self.alignment_manager.update_alignment(
            self.current_file_a, 
            self.current_file_b, 
            offset_x, 
            offset_y, 
            moving_image
        )
        
        QMessageBox.information(self, "Сохранено", "Настройки смещения сохранены")
    
    def auto_align(self):
        """Выполняет автоматическое выравнивание"""
        if not self.current_file_a or not self.current_file_b:
            return
        
        try:
            moving_image = "A" if self.moving_combo.currentIndex() == 1 else "B"
            offset_x, offset_y = self.alignment_manager.auto_align_images(
                self.current_file_a, 
                self.current_file_b, 
                moving_image
            )
            
            self.x_spin.setValue(offset_x)
            self.y_spin.setValue(offset_y)
            
            QMessageBox.information(self, "Успешно", 
                                  f"Автоматическое выравнивание выполнено\n"
                                  f"Смещение: X={offset_x}, Y={offset_y}")
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось выполнить автоматическое выравнивание:\n{str(e)}")
    
    def on_point_selection_mode_toggled(self, checked: bool):
        """Обработчик переключения режима выбора точек"""
        self.points_widget_a.setVisible(checked)
        self.points_widget_b.setVisible(checked)
        self.calculate_from_points_btn.setEnabled(checked)
        
        if checked:
            # Очищаем точки при включении режима
            self.points_widget_a.clearPoints()
            self.points_widget_b.clearPoints()
    
    def on_point_selected_a(self, x: int, y: int):
        """Обработчик выбора точки на изображении A"""
        # Проверяем, что количество точек совпадает
        points_a = self.points_widget_a.getPoints()
        points_b = self.points_widget_b.getPoints()
        
        if len(points_a) > len(points_b):
            # Предлагаем выбрать соответствующую точку на B
            QMessageBox.information(self, "Выбор точки", 
                                  f"Теперь выберите соответствующую точку на изображении B")
    
    def on_point_selected_b(self, x: int, y: int):
        """Обработчик выбора точки на изображении B"""
        # Проверяем, что количество точек совпадает
        points_a = self.points_widget_a.getPoints()
        points_b = self.points_widget_b.getPoints()
        
        if len(points_b) > len(points_a):
            # Предлагаем выбрать соответствующую точку на A
            QMessageBox.information(self, "Выбор точки", 
                                  f"Теперь выберите соответствующую точку на изображении A")
    
    def calculate_from_points(self):
        """Рассчитывает смещение на основе выбранных точек"""
        points_a = self.points_widget_a.getPoints()
        points_b = self.points_widget_b.getPoints()
        
        if len(points_a) != len(points_b):
            QMessageBox.warning(self, "Ошибка", 
                              f"Количество точек не совпадает: A={len(points_a)}, B={len(points_b)}")
            return
        
        if len(points_a) < 2:
            QMessageBox.warning(self, "Ошибка", "Необходимо выбрать минимум 2 соответствующие точки")
            return
        
        try:
            offset_x, offset_y = self.alignment_manager.calculate_alignment_from_points(
                self.current_file_a, 
                self.current_file_b, 
                points_a, 
                points_b
            )
            
            self.x_spin.setValue(offset_x)
            self.y_spin.setValue(offset_y)
            
            QMessageBox.information(self, "Успешно", 
                                  f"Смещение рассчитано по точкам\n"
                                  f"Смещение: X={offset_x}, Y={offset_y}")
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось рассчитать смещение:\n{str(e)}") 

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QPoint

class SliderReveal(QWidget):
    def __init__(self, pix_old: QPixmap, pix_new: QPixmap, parent=None):
        super().__init__(parent)
        self.pix_old = pix_old
        self.pix_new = pix_new
        self.split_x = self.width() // 2
        self.setMouseTracking(True)
        self._drag = False
        self._pan = False
        self._last_pos = QPoint()
        self.offset = QPoint(0, 0)
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 8.0

    def resizeEvent(self, _):
        self.split_x = self.width() // 2
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if abs(e.x() - self.split_x) < 8:
                self._drag = True
            else:
                self._pan = True
            self._last_pos = e.pos()

    def mouseMoveEvent(self, e):
        if self._drag:
            self.split_x = max(0, min(e.x(), self.width()))
            self.update()
        elif self._pan:
            delta = e.pos() - self._last_pos
            self.offset += delta
            self._last_pos = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = False
            self._pan = False

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            angle = e.angleDelta().y()
            factor = 1.2 if angle > 0 else 1/1.2
            old_scale = self.scale
            self.scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
            # Центрируем zoom относительно курсора
            mouse_pos = e.pos()
            rel = mouse_pos - self.offset
            relf = rel * self.scale / old_scale
            self.offset = mouse_pos - QPoint(int(relf.x()), int(relf.y()))
            self.update()

    def setPixmaps(self, pix_old: QPixmap, pix_new: QPixmap):
        self.pix_old = pix_old
        self.pix_new = pix_new
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Применяем трансформацию (offset, scale)
        qp.translate(self.offset)
        qp.scale(self.scale, self.scale)
        
        # Рисуем новое изображение полностью
        if not self.pix_new.isNull():
            qp.drawPixmap(0, 0, self.pix_new)
        
        # Клипируем старое по split_x
        if not self.pix_old.isNull():
            qp.save()
            # Вычисляем позицию слайдера в координатах изображения
            split_pos_scaled = (self.split_x - self.offset.x()) / self.scale
            clip_y = -self.offset.y() / self.scale
            clip_h = self.height() / self.scale
            qp.setClipRect(0, int(clip_y), int(split_pos_scaled), int(clip_h))
            qp.drawPixmap(0, 0, self.pix_old)
            qp.restore()
        
        qp.resetTransform()
        
        # Рисуем handle
        pen = QPen(QColor('#0078D7'))
        pen.setWidth(3)
        qp.setPen(pen)
        qp.drawLine(self.split_x, 0, self.split_x, self.height())
        qp.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_0 and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.reset_view()
        else:
            super().keyPressEvent(event)

    def reset_view(self):
        self.offset = QPoint(0, 0)
        self.scale = 1.0
        self.update() 